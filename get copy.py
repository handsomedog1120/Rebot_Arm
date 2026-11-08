#!/usr/bin/env python3
"""
RebotArmEndPose 交互控制示例（轨迹规划模式）。
输入: x y z [roll pitch yaw] [duration]  目标末端位置（米 / 弧度 / 秒）
      g <pos>                            设置夹爪目标位置

RebotArmEndPose interactive control example (trajectory planning mode).
Input: x y z [roll pitch yaw] [duration]  target end-effector pose (meters / radians / seconds)
       g <pos>                            set gripper target position

用法 / Usage:
    python example/8_arm_traj_control.py

退出 / Exit: q / quit / exit
状态 / State: state, end_state
"""
import json
import os
import sys
import threading
import time
import traceback


from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import ( 
    load_robot_model,
    get_end_effector_frame_id,
    compute_fk,
    pos_rot_to_se3,
    pad_q_for_model,
    solve_ik,
    solve_ik_with_retry,
    IKSolverParams,
    robot_model as _rmod,
)

import yaml
import pinocchio as pin
import numpy as np


_model = None
_lock = threading.Lock()

class rebot_control():
    global _model, _lock
    def __init__(self):
        self.rebotarm = RebotArm(hw_yaml = "rebotarm_rs.yaml")
        self.ctrl = RebotArmEndPose(self.rebotarm, arm_control_mode="mit")
        _model = load_robot_model()

        self.duration = 3.0

    def _log(self, msg):
        print(msg, file=sys.stderr, flush=True)


    
    def normal_to_rpy(nx, ny, nz, roll=0.0):
        """由表面法向量构造末端姿态的 roll/pitch/yaw (rad).

        约定与 arm_server 的 _rot_from_normal 完全一致:
          - 夹爪接近轴 = 末端 local X = 输入法向量方向
          - 腕部自转 roll 绕接近轴 (因法向量本身无自转信息, 这里用传入的默认值)

        参数:
            nx, ny, nz: 法向量 (任意长度, 会归一化)
            roll:       绕接近轴的自转角 (rad), 默认 0; 若要沿用 arm_server
                        多横滚枚举的某一个 (如 +π/2), 这里填进去

        返回:
            [roll, pitch, yaw]  单位 rad
        """
        x = np.array([nx, ny, nz], dtype=float)
        n = np.linalg.norm(x)
        if n < 1e-6:
            # 退化: 法向量零, 返回默认夹爪朝下姿态的 rpy
            return list(pin.rpy.matrixToRpy(
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)))
        x = x / n

        # z = 世界 Z 投影到垂直于 x 的平面
        z0 = np.array([0.0, 0.0, 1.0])
        z = z0 - (z0 @ x) * x
        if np.linalg.norm(z) < 1e-6:
            z0 = np.array([0.0, 1.0, 0.0])
            z = z0 - (z0 @ x) * x
        z = z / np.linalg.norm(z)

        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        R = np.column_stack([x, y, z])          # 基础矩阵 (roll=0)

        # 绕接近轴 (R 第 1 列) 滚 roll 角
        if roll != 0.0:
            c, s = np.cos(roll), np.sin(roll)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            R = R @ Rx

        rpy = pin.rpy.matrixToRpy(R)
        return [float(rpy[0]), float(rpy[1]), float(rpy[2])]




    def move_traj(self, x, y, z, normal=None):
        if normal is None:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        else:
            roll, pitch, yaw = self.normal_to_rpy(*normal)
        ok = self.ctrl.move_to_traj(
                    x=x, y=y, z=z,
                    roll=roll, pitch=pitch, yaw=yaw,
                    duration=self.duration,
                )




    def _handle(self,cmd) :
        c = cmd.get("cmd")

        x = cmd.get("x")
        y = cmd.get("y")
        z = cmd.get("z")    

        normal = cmd.get("normal")

        _lock.acquire()
        try:
            if c == "connect":
                self.ctrl.start()
                return {"ok": True}
            
            if self.ctrl is None:
                return {"ok": False, "err": "not connected"}
            
            if c == "enable":
                
                self._log("[enable] enable_all done"); return {"ok": True}
            

            if c == "disable":
                self.rebotarm.disable_all(); return {"ok": True}
            if c == "estop":
                self.rebotarm.estop(); return {"ok": True}
            if c == "get_state":
                q, _, _ = self.rebotarm.get_state()
                joints = [float(v) for v in q[:self.ctrl._n]]
                g = float(q[self.ctrl._n]) if self.rebotarm.has_gripper and len(q) > self.ctrl._n else 0.0
                # traj_active: 服务器端轨迹是否仍在执行(UI 阻塞等待的完成判据)
                return {"ok": True, "joints": joints, "gripper": g}
            if c == "can_ping":
                # ★ CAN往返探测: 读joint1 mechPos(0x7019)等真实回包.
                #   PCAN适配器"冻结"(dmesg: Rx urb aborted -32)时 netdev 仍 up、
                #   发送不报错, 但 CAN 回包全丢 —— get_state 读的是缓存,
                #   只有这里能探测出"收不到"的真死.
                if self.rebotarm is None or not getattr(self.rebotarm, "_connected", False):
                    return {"ok": False, "err": "not connected"}
                try:
                    jn = self.rebotarm._all_joints[0].name
                    self.rebotarm._motor_map[jn].robstride_get_param_f32(0x7019, timeout_ms=600)
                    return {"ok": True}
                except Exception as e:
                    return {"ok": False, "err": "can rx dead: %s" % e}
            if c == "get_pose":
                # 当前关节 -> 末端位姿 (FK), 供现场标定预设点用
                q, _, _ = self.rebotarm.get_state()
                q_pad = pad_q_for_model(_model, q[:self.ctrl._n], self.ctrl._n)
                pos, rot, _ = compute_fk(_model, q_pad)
                rpy = pin.rpy.matrixToRpy(np.array(rot)).tolist()
                return {"ok": True, "pos": [float(v) for v in pos],
                        "rpy": [float(v) for v in rpy],
                        "joints": [float(v) for v in q[:self.ctrl._n]]}


            if c == "move_traj":
                self.move_traj(x,y,z, normal=normal)

            if c == "set_gripper":
                self.ctrl.set_gripper_target(float(cmd["pos"]))
                return {"ok": True}
            
            if c == "home":
                self.ctrl.safe_home(); return {"ok": True}
            if c == "shutdown":
                # ★ 2026-08-31: 关UI时直接disable(不回零) — 原版end()里safe_home
                #   先回零再断开(10-20秒), UI等30s超时或进程被强杀时电机保持
                #   使能态 → PCAN灯常亮. 现在改立即停控制循环+disable+断开, 2秒内完成.
                try:
                    self.ctrl.stop_control_loop()
                    self.ctrl.disable_all()
                    self._log("[shutdown] 控制循环已停+全部电机已去使能")
                except Exception as e:
                    self._log("[shutdown] disable异常(仍继续断开): %s" % e)
                try:
                    self.rebotarm.disconnect()   # 内部也做 disable_all + ctrl.close
                    self._log("[shutdown] CAN断开完成")
                except Exception as e:
                    self._log("[shutdown] disconnect异常: %s" % e)
                return {"ok": True}
            return {"ok": False, "err": "unknown cmd: %s" % c}

        except Exception as e:
            self._log("EXC:%s\n%s" % (e, traceback.format_exc()))
        finally:
            _lock.release()



def main() -> None:
    _arm = rebot_control()
    _arm._log("rebot_arm_server ready (rebot py3.12) pid=%d" % os.getpid())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "err": "bad json: %s" % e}), flush=True)
            continue
        # ★ set_joints 高频指令不回包: UI 端 fire-and-forget(卡顿修复)的配套.
        #   若回包, 响应会积压在 stdout 管道被后续命令误读(协议错位).
        if cmd.get("cmd") == "set_joints":
            _arm._handle(cmd)
            # 控制循环死亡检测: 线程崩了但进程活着 = "连接了但发不出指令"(灯常亮假象),
            # 立即报错退出让 UI 触发重启自愈, 不再静默装死
            if _arm._ctrl is not None and getattr(_arm._ctrl, "_running", False) and not _arm.control_loop_active:
                _arm._log("[FATAL] 控制循环线程已死亡(内部异常), arm_server 退出待UI重启")
                break
            continue
        resp = _arm._handle(cmd)
        print(json.dumps(resp), flush=True)
        if cmd.get("cmd") == "shutdown":
            break
    # ★ stdin EOF (UI 崩溃/被杀): 不让电机瞬间掉电失控 ——
    #   保持当前目标发 2 秒 MIT 保持指令, 然后安全断开(断开内部会 disable).
    #   给用户 2 秒窗口期反应 + 电机受控泄力而非猛掉.
    _arm._log("[EOF] stdin 关闭(UI退出?), 进入 2 秒保持后安全退出")
    if _arm._ctrl is not None:
        try:
            import threading as _th
            _t = _th.Timer(2.0, lambda: (_arm.disconnect() if _arm else None))
            _t.daemon = True
            _t.start()
            _t.join()
        except Exception as e:
            _arm._log("[EOF] 安全退出异常: %s" % e)
    _arm._log("rebot_arm_server exit")


if __name__ == "__main__":
    main()

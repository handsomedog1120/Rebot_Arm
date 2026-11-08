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


# ---- reBotArm_control_py 路径 ----
# ★ 必须指向外层 repo 目录(内层才是带 __init__.py 的真包):
#   若指到 repo 的上一层, 外层 reBotArm_control_py/ 会被当成命名空间包,
#   没有 actuator 子模块 -> 启动即 ModuleNotFoundError -> UI"启动无反应"
REPO = os.path.expanduser("~/forensic_ws/src/reBot/reBotArm_control_py")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.dynamics import compute_generalized_gravity
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
_ik_data = None   # ★ IK专用data: pinocchio Data 非线程安全, 与控制循环 _loop_cb
                 #   的重力前馈共用 data 会数据竞争->段错误->arm_server静默死亡
                 #   (旧服务器"动作一开始就整体失能"的根因)
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────
# ★ 2026-11-05 monkey-patch RebotArmEndPose: 修复 _send_loop 竞态 Bug
# 现象: 涂抹/粘取动作期间后端 move_traj 连发(30ms间隔), 机械臂停住
# 根因链:
#   1. 后端 move_traj B 持锁执行 set _stop_send + join(timeout=5.0) 等旧轨迹 A
#   2. join timeout 后 clear _stop_send + self._moving=True + B.start()  + 释放锁
#   3. 旧轨迹 A 还在跑(sleep 中), 醒来后 _stop_send 已被 clear → 继续
#   4. A 跑完剩余循环 → 末尾无条件 self._moving = False → 覆盖 B 的 True!
#   5. UI 轮询 get_state → traj_active=False → 误判 B 完成 → 发下一个
#   6. 后端收到下一个 move_traj → 又 set _stop_send 取消当前 B → 雪崩连发
# 第一次修复 Bug: 调 _orig_send_loop 后再检查 epoch — 但 orig 内部已经
#   self._moving=False 无条件清了, 返回后再检查已经晚了.
# 本次修复: 完全替换 _send_loop, 只在自己仍是当前 epoch 时才清 _moving;
#   move_to_traj 每次 start 新轨迹 bump epoch, 旧线程退出时 epoch 不匹配
#   就跳过不清 — B 的 True 状态得以保留, UI 轮询看到 traj_active=True,
#   不会误判完成.
# ─────────────────────────────────────────────────────────────────────────

def _patched_send_loop(self, duration):
    """完全替换原始 _send_loop — epoch 机制保护 _moving 不被旧线程覆盖."""
    _my_epoch = getattr(self, '_traj_epoch', 0)
    n = len(self._traj)
    interval = duration / n if n > 0 else self._dt
    for i in range(n):
        if self._stop_send.is_set():
            return           # ★ 被新轨迹取消: 直接 return, 不清 _moving
        self._q_target[:] = self._traj[i]
        time.sleep(interval)
    # ★ 跑完正常退出: 只在自己 epoch 没被覆盖时才清 _moving
    if getattr(self, '_traj_epoch', -1) == _my_epoch:
        self._moving = False


def _patched_move_to_traj(self, *args, **kwargs):
    # 每次 start 新轨迹时 bump epoch —— 让旧线程退出时能识别自己已被替代
    if not hasattr(self, '_traj_epoch'):
        self._traj_epoch = 0
    self._traj_epoch += 1
    return RebotArmEndPose.move_to_traj.__wrapped__(self, *args, **kwargs)


# ★ patch 注册: 用 __wrapped__ 保留原始引用(供 move_to_traj 内部调用)
_orig_move_to_traj = RebotArmEndPose.move_to_traj
_patched_move_to_traj.__wrapped__ = _orig_move_to_traj

RebotArmEndPose._send_loop = _patched_send_loop
RebotArmEndPose.move_to_traj = _patched_move_to_traj


class rebot_control():
    def __init__(self):
        # ★ global 必须在方法内声明: class 体的 global 不作用于方法,
        #   否则 _model 只是局部变量, 模块级 _model 恒为 None
        #   -> get_pose/solve_ik 运行时 'NoneType' has no attribute 'nq'
        global _model
        self.rebotarm = RebotArm(hw_yaml = "rebotarm_rs.yaml")
        self.ctrl = RebotArmEndPose(self.rebotarm, arm_control_mode="mit")
        _model = load_robot_model()

        self.duration = 2.0
        self._grip_ramp_stop = threading.Event()   # 夹爪斜坡线程打断标志
        # ★ 末段收敛助推状态(2026-10-25): 见 _loop_cb_boost
        self._boost_kp_on = False
        self._boost_stuck_t0 = None
        self._boost_deadline = None
        self._boost_tries = 0
        self._boost_prev_qt = None
        # ★ 2026-11-05 放弃闩锁: 3次助推失败后不再无限重试.
        #   旧逻辑"放弃"后下一段运动 moved=True 即清零计次 → J1 残差
        #   ~0.02rad 推不动(kp=150 都不行) → 助推每 ~10s 重来一轮, 每轮
        #   爬坡到 kp=150 持续数秒 ~3N·m 大电流 → 48V 母线电流尖峰 →
        #   共线关节(J1/J2/J3 同分支)+重载 J4 过流静默失能 →"动作做完
        #   就掉线". 闩锁仅在关节真正收敛(err<ERR_LO)或 _boost_reset
        #   (连接/使能/恢复/新 move_traj 显式清)时解除.
        self._boost_gave_up = False
        # ★ 电机健康检查(2026-10-28): 1Hz 定期巡检 + 助推放弃后立即巡检;
        #   固件瞬时过流保护会关电机输出但 status_code 仍为 0(瞬态 fault),
        #   表现为 torq=0 + 位置下滑.
        # ★ 2026-10-30 用户安全要求: 检测到失能【只报警+停轨迹, 禁止自动
        #   clear_error+enable】—— 电机在臂下垂/有人靠近时突然重新出力会
        #   猛拽伤人. 恢复只能由用户确认安全后手动点「启动机械臂」
        #   (_connect/_recover), 这两条路径会清除 _motor_fault 重新启用巡检.
        self._hbeat_last = 0.0
        self._motor_fault = False
        self._fault_streak = {}   # 逐关节连续失能判定计数(见 _health_check)
        # ★ 2026-10-31: 电机"应处于使能"标志 —— 用户主动 disable/estop 时臂
        #   也会下垂(torq≈0 + 偏离目标), 若无此门会被巡检误报成"失能".
        self._motors_enabled = False

    def _log(self, msg):
        print(msg, file=sys.stderr, flush=True)



    def _gripper_ramp(self, tgt):
        """★ 夹爪目标柔顺渐进逼近(复刻旧服务器 GRIP_VMAX 行为):
        直写大步长目标 —— MIT kp=10 下 4+rad 的阶跃指令瞬间满力矩,
        电机过流保护性锁死, 表现就是"点设定/复位0 夹爪不动"(标定的小步匍匐能动).
        新目标到达时打断进行中的旧 ramp(最新目标优先); 小步(<单拍步进)立即直写,
        与 UI 标定匍匐的连续小步指令兼容."""
        self._grip_ramp_stop.set()          # 打断旧 ramp
        stop = threading.Event()
        self._grip_ramp_stop = stop
        def _run():
            step = 1.50 * 0.05              # ★ 1.50 rad/s @ 50ms ≈ 23mm/s (2026-11-05 提速: 原 0.60rad/s)
            cur = float(self.ctrl._gripper_target)
            while not stop.is_set() and abs(tgt - cur) > step:
                cur += step if tgt > cur else -step
                self.ctrl._gripper_target = cur
                time.sleep(0.05)
            if not stop.is_set():
                self.ctrl._gripper_target = tgt
        threading.Thread(target=_run, daemon=True).start()

    def _auto_duration(self, x, y, z):
        """★ 距离自适应时长(2026-10-18): 近移用时短/远移用时长, 速度感均匀.
        起点末端 = FK(当前指令目标 _q_target, 与IK种子一致, 不做CAN往返),
        T = clamp(d / V_CART + T_BASE):
          V_CART=0.09 m/s 末端均速
          T_BASE=1.2s     起停加减速基础时间
          T_MIN=2.0s      过短会让相邻轨迹点跳变过大, 电机猛追过冲
          T_MAX=10.0s     大转移封顶, 防慢吞吞
        ★ 2026-10-30 降加速度: 原 0.16 m/s + 0.8s 起停, 峰值加速度偏高 ——
          运动瞬间 J1/J2/J3(RS-06)三台大电机同时吃峰值电流, 触发驱动板
          保护集体关输出("同时掉线"). min-jerk 下峰值加速度 ∝ 1/T²,
          时长放长约 1.4 倍 → 加速度约降一半, 电流尖峰同步减半.
          注: 轨迹加速度由时长决定, 无独立 ACC_MAX 常量
          (rebot_arm_server.py 里的 ACC_MAX 属旧后端, UI 不加载)."""
        global _ik_data
        try:
            n = self.ctrl._n
            q = pad_q_for_model(_model, np.asarray(
                self.ctrl._q_target[:n], dtype=float), n)
            if _ik_data is None:
                _ik_data = _model.createData()
            pin.forwardKinematics(_model, _ik_data, q)
            pin.updateFramePlacements(_model, _ik_data)
            p = np.asarray(_ik_data.oMf[self.ctrl._end_frame_id].translation)
            d = float(np.linalg.norm(np.array([x, y, z], dtype=float) - p))
        except Exception as e:
            self._log("[auto_dur] FK失败(回退默认%.1fs): %s" % (self.duration, e))
            return float(self.duration)
        T = d / 0.32 + 0.8               # ★ 2026-10-30: 0.16+0.8 → 0.09+1.2 (降加速度)
        T = float(min(max(T, 2.0), 10.0))
        self._log("[auto_dur] 距离 %.3f m -> 时长 %.1f s" % (d, T))
        return T

    def _stop_traj(self):
        """★ 取消进行中的服务器端轨迹(新运动/急停/失能/恢复前调用):
        停 _send_loop 线程并清 _moving/_traj —— 否则 estop 后重新使能时,
        旧轨迹残段会把臂继续拽走."""
        self.ctrl._stop_send.set()
        t = self.ctrl._send_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self.ctrl._moving = False
        self.ctrl._traj = []

    # ── ★ 末段收敛助推(2026-10-25): J1 ±0.06 回零卡死修复 ──────────────────
    #   根因: 所有运动路径(流式set_joints/safe_home/move_traj)都是纯位置式
    #   MIT(_qd_target 全代码从未被写入, 无速度前馈), 电机出力上限 =
    #   kp×误差 = 25×0.06 ≈ 1.5N·m; J1 扛整臂静摩擦 ≥ 此值 → 误差进入
    #   ±0.06 带内永远推不动("返回初始位置"卡死根因, 动作收尾残差同理).
    #   助推: 目标静止且某关节残差卡在带内持续0.5s = 静摩擦卡滞, 临时把
    #   该关节 kp 提到 60(峰值力矩 60×0.08=4.8, 安全量级)冲破摩擦;
    #   收敛/超时/目标重新运动 即恢复 25. 重力门限排除重力下垂关节
    #   (j2-j5 满载下垂 0.06~0.075rad 恰在带内, 但那是重力不是摩擦,
    #   提kp只会引起"抬高-下垂"往复), 只助推重力力矩可忽略的关节
    #   (J1轴竖直重力≈0 / 收拢姿态下的轻载关节).
    _BOOST_KP = 100.0       # ★ 助推增益(2026-10-30: 150→100, 用户反馈150太猛):
    _BOOST_KP_MAX = 150.0   # ★ 2026-10-31 爬坡上限: 实测 kp=100 推不动 J1
                            #   (err 0.023rad 纹丝不动=静摩擦>2.3N·m), 恒定150
                            #   用户嫌太猛 → 助推中从 100 起每 0.4s +25 爬坡,
                            #   以刚好破摩擦的最小力矩起跳减小冲击, 上限175
                            #   J1常态60, 助推100仍有明显增量可推过静摩擦,
                            #   峰值力矩降到100×0.12=12, 更温和.
                            #   (历史: 60→150 是因助推60对J1零增量; 150≈原yaml
                            #   硬增益量级, 被固件力矩限幅钳住, 护栏内短促出力)
    _BOOST_ERR_LO = 0.0025  # 收敛判据(rad) ★ 2026-10-31 0.005→0.0025:
                            #   验收实测 J1 残差 0.28°(0.0049rad) 恰卡在旧
                            #   下沿之外助推不启动, 放宽到 0.14° 让爬坡收尾
    _BOOST_ERR_HI = 0.12    # ★ 静摩擦带上限(0.08→0.12): "差一点"若超过0.08,
                            #   助推根本不触发, 只剩常态kp硬扛. J1扛整臂,
                            #   残差可能停在带外. 峰值力矩100×0.12=12被固件限幅
    _BOOST_STUCK_S = 0.5    # 卡滞判定时长(s)
    _BOOST_TIMEOUT_S = 3.0  # 单次助推超时(s)
    _BOOST_MAX_TRIES = 3    # 连续失败上限(防堵转反复出力发热)
    _BOOST_GRAV_GATE = 0.5  # 重力门限(N·m): 关节重力小于此=残差是摩擦
    # ★ 助推范围(2026-10-28 收缩到只剩 J1):
    #   之前 J3(索引2)、J5(索引4)、J6(索引5) 都会被助推, 其中 J3 kp=150
    #   + 大惯量 → 瞬态过流关输出(torq=0), 机械臂移动时频繁掉线. 用户明确
    #   要求除 J1 外一律不助推. J1 轴竖直重力≈0, kp=150 安全(历史已验证).
    _BOOST_ONLY_INDEX = {0}   # 仅索引 0 = J1(底座旋转) 允许助推
    _BOOST_GRAV_EXEMPT = set()

    # ★ 2026-10-31 用户决定: 撤掉"目标点慢速游走"(方案A —— 实测中关节跟不上
    #   游走目标导致拖跑/超调), 改用方案B: 直接提 J3 常态 kp 25→40,
    #   稳态误差=前馈残差/kp 同比缩小 ~40%; 配合 J3 kd 2→3 压住水平
    #   大惯量下的振荡(10-28 回退记录的风险项).

    # ★ 关节常态软增益(2026-10-25): J1 扛整臂静摩擦大, kp 单独提高到 60
    #   (25 时 ±0.06 带内出力 1.5N·m 推不动静摩擦; 60 仍推不动的残差
    #   由末段助推 kp=150 收尾, 见 _BOOST_KP).
    #   其余关节保持 25 —— 所有"恢复软增益"路径必须走 _apply_arm_kp,
    #   不得再写 _mit_kp[:]=25(会把 J1 打回 25).
    _ARM_KP_BASE = 25.0
    _ARM_KP_J1 = 50.0
    _ARM_KP_J3 = 50.0   # ★ 2026-10-31 方案B(用户选定, 替代目标游走): J3(肘部
                        #   俯仰,索引2) 水平伸臂重力负载最大, kp=25 时稳态残差
                        #   =前馈残差/kp 到不了目标; 提 kp 直接同比缩小残差
                        #   (25→40 ≈ 缩小38%). 振荡风险由 J3 专属 kd=3 压住
                        #   (见 _ARM_KD_J3). 若仍振荡回退到 32~35.
    _ARM_KP_J4 = 25.0   # ★ 2026-10-25: J4(腕部俯仰,索引3) x=0.217臂展边缘
                        #   末段稳态误差收敛慢("到位停住后又补走一小段"),
                        #   提常态kp直接减小稳态误差; 与J1同思路.
                        #   50是阻尼比边界, 若出现振荡需配合提kd(当前3)
    _ARM_KD = 2.0       # ★ 2026-10-28: 阻尼统一 2→3, J3 水平大惯量下 ζ
                        #   从 0.65 提升到 ~0.85, 压住周期性摆动; 收拢小惯量
                        #   下 ζ≈1.1 过阻尼可接受(响应慢一点但不抖).
    _ARM_KD_J3 = 3.0    # ★ 2026-10-31 方案B配套: J3 大惯量+高kp 需更高阻尼,
                        #   kd 逐关节独立(_mit_kd 数组), 只提 J3 不影响其余关节

    # ★ 到位闭环一次性补偿(2026-10-31 方案A, 用户选定): 纯位置式MIT的稳态
    #   残差随姿态变化(J3低位实测-1.1°→末端7mm), 固定前馈系数无法全姿态
    #   归零(高位刚好/低位欠补). 停稳后实测末端偏差超阈值 → 以"目标+误差
    #   向量"补发一条短轨迹, 最多2次收敛. 详见 _arrival_compensate.
    _COMP_ERR_THRESH_MM = 4.0   # 触发补偿的末端偏差阈值(mm)
    _COMP_ERR_MAX_MM = 30.0     # 超过视为异常量级(失能/跳解), 不硬补
    _COMP_MAX_ITER = 2          # 最大补偿次数

    def _apply_arm_kp(self):
        """恢复常态软增益: BASE=25, J1=50/J3=25/J4=25(索引 0/2/3)."""
        ag = self.ctrl._arm_group
        if ag is not None:
            ag._mit_kp[:] = self._ARM_KP_BASE
            ag._mit_kp[0] = self._ARM_KP_J1
            ag._mit_kp[2] = self._ARM_KP_J3
            ag._mit_kp[3] = self._ARM_KP_J4

    def _boost_reset(self):
        """助推状态清零+增益还原(连接/恢复/急停/失能时调用).
        ★ 也清放弃闩锁 — 这些是"用户显式重来"的时机, 允许助推重新尝试."""
        if self._boost_kp_on:
            self._boost_kp_on = False
            self._apply_arm_kp()
        self._boost_stuck_t0 = None
        self._boost_deadline = None
        self._boost_tries = 0
        self._boost_prev_qt = None
        self._boost_gave_up = False

    def _boost_halt(self):
        """★ 2026-11-05 只停进行中的助推并还原增益, 保留失败计次/放弃闩锁.
        供 move_traj 起步前调用 — 轨迹起步只需保证不残留 boost kp(顿挫
        修复), 不应重置"放弃"决策, 否则动作里每段轨迹结束都会重新进入
        助推重试循环(无限电流尖峰的根因)."""
        if self._boost_kp_on:
            self._boost_kp_on = False
            self._apply_arm_kp()
        self._boost_stuck_t0 = None
        self._boost_deadline = None

    def _health_check(self):
        """★ 电机失能检测【只报警, 不停轨迹, 不自动恢复】(2026-10-30):
        RobStride 固件过流/跟随误差保护后会关电机输出, 但 status_code 仍为 0,
        表现为 torq≈0 + 位置被重力拖走.

        ★★ 判据必须严格限定在【指令静止(保持/待命)】时 —— 运动过程中
        "vel>0.1 且 torq≈0" 完全正常(J1 轴竖直/腕部轻载/夹爪 kp=10, 净力矩
        本来就≈0), 早先版本据此判失能并 _stop_traj(), 结果一运动就误报停机
        (日志里连 joint5/joint6/gripper 都被报"失能", 全是误报).
        现在三个条件同时满足才判失能, 且需连续 2 次(2s)确认:
          ① 无进行中的轨迹(_moving=False), 指令目标静止, 且电机应处于使能
             (用户主动 disable/estop 时臂下垂不算失能, _motors_enabled 门)
          ② 位置偏离目标 >0.05 rad, 且【未在向目标收敛】:
             正在漂走(err*vel<0) 或 垂定静止(vel≈0) 都算 ——
             ★ 2026-10-31 漏检修复: 旧判据要求 |vel|>0.05(正在漂走), 但
               固件锁死后臂垂到位即静止(vel≈0), 永远凑不齐条件, J2/J3
               锁死一次都没报出来. 使能电机在 err=0.05 时 MIT 出力
               kp×err≥30×0.05=1.5N·m, 所以 torq<0.3+err>0.05+不收敛
               三者同真必然是输出被关.
          ③ torq <0.3 N·m(电机确实没出力); 夹爪不参与(无重力保持语义)
        报警后置 _motor_fault 停止巡检(防刷日志), 恢复只由用户点「启动机械臂」."""
        if self._motor_fault or not self._motors_enabled:
            return   # 已报警等手动恢复 / 用户主动去使能: 都不巡检(省CAN流量)
        arm = self.rebotarm
        if arm is None or not getattr(arm, "_connected", False):
            return
        # ★ 运动过程中一律不判定(误报根源), 同时清空连续计数
        _thr = getattr(self.ctrl, "_send_thread", None)
        if getattr(self.ctrl, "_moving", False) or (_thr is not None and _thr.is_alive()):
            self._fault_streak.clear()
            return
        # ★ 只读缓存, 不做 request_feedback/poll 往返: 本函数运行在 250Hz
        #   控制循环线程里, 一旦某电机不回帧, 逐电机往返会把 timeout(2s)
        #   串行叠加 → 控制循环卡死. MIT 指令在 250Hz 下发, 电机每帧都回,
        #   缓存本身就是新鲜的.
        n = self.ctrl._n
        faulted = []
        for i, jc in enumerate(arm._all_joints[:n]):
            if "gripper" in jc.name.lower():
                continue
            try:
                st = arm._motor_map[jc.name].get_state()
            except Exception:
                continue
            if st is None or st.status_code != 0:
                self._fault_streak.pop(jc.name, None)
                continue
            err = float(self.ctrl._q_target[i]) - float(st.pos)
            # 正在向目标收敛(err*vel>0 且速度够) = 正常, 其余(漂走/垂定静止)
            # 只要"偏离+无力矩"即判输出被关 —— 含固件锁死后垂定(vel≈0)情形
            converging = (err * float(st.vel) > 0 and abs(st.vel) > 0.05)
            droop = (abs(err) > 0.05 and abs(st.torq) < 0.3 and not converging)
            if droop:
                k = self._fault_streak.get(jc.name, 0) + 1
                self._fault_streak[jc.name] = k
                if k >= 2:   # 连续 2 次(1Hz)确认, 排除单拍噪声
                    faulted.append("%s(torq=%.2f, vel=%.2f, err=%.3f)" % (
                        jc.name, st.torq, st.vel, err))
            else:
                self._fault_streak.pop(jc.name, None)
        if faulted:
            self._motor_fault = True
            self._log("[health] ★★ 检测到电机失能: %s — 【不自动恢复】"
                      "(防伤人). 请确认臂周围安全后, 手动点「启动机械臂」恢复."
                      % ", ".join(faulted))

    def _loop_cb_boost(self, arm, dt):
        """★ 250Hz 包装回调: 先做末段卡滞检测/增益助推, 再调原 _loop_cb 发MIT.
        重力计算与 _loop_cb 用同一个 ctrl._data —— 本包装与 _loop_cb 同线程
        顺序执行, 无 Data 竞争; 读反馈走缓存(request_feedback=False), 零CAN开销."""
        try:
            ag = self.ctrl._arm_group
            n = self.ctrl._n
            if ag is not None and n > 0:
                qt = np.asarray(self.ctrl._q_target[:n], dtype=float)
                moved = (self._boost_prev_qt is not None and
                         qt.shape == self._boost_prev_qt.shape and
                         float(np.max(np.abs(qt - self._boost_prev_qt))) > 1e-4)
                self._boost_prev_qt = qt.copy()
                q_now = np.asarray(
                    ag.get_positions(request_feedback=False), dtype=float)
                now = time.monotonic()
                if q_now.size == n:
                    err = np.abs(qt - q_now)
                    # 重力门限: 残差是摩擦(可助推)还是重力下垂(不可助推)
                    try:
                        q_pad = pad_q_for_model(self.ctrl._model, q_now, n)
                        g = np.abs(compute_generalized_gravity(
                            self.ctrl._model, q_pad, self.ctrl._data)[:n])
                    except Exception:
                        g = None
                    if g is not None:
                        # ★ 助推硬范围 + 残差带 + 重力门限三重过滤:
                        #   1. _BOOST_ONLY_INDEX = {0} → 除 J1 外一律不助推(用户要求)
                        #   2. 残差落在静摩擦带内(0.005~0.12 rad)
                        #   3. 重力门限: 重力可忽略或显式豁免
                        elig = [j for j in range(n)
                                if (j in self._BOOST_ONLY_INDEX
                                    and self._BOOST_ERR_LO < err[j] < self._BOOST_ERR_HI
                                    and (g[j] < self._BOOST_GRAV_GATE
                                         or j in self._BOOST_GRAV_EXEMPT))]
                        if moved:
                            # 目标在动(流式/轨迹中): 不助推; 新指令打断进行中的助推
                            # ★ 2026-11-05: 不再清零 tries — 旧逻辑"下段运动即
                            #   清零"使放弃后助推无限重试(kp=150 电流尖峰
                            #   反复冲击48V母线), 是"动作做完就掉线"根因.
                            #   tries/gave_up 只由真正收敛或 _boost_reset 清.
                            if self._boost_kp_on:
                                self._boost_kp_on = False
                                self._apply_arm_kp()
                            self._boost_stuck_t0 = None
                        elif self._boost_kp_on:
                            if not elig:
                                # 全部收敛/离开带: 恢复常态增益, 计次清零
                                self._boost_kp_on = False
                                self._apply_arm_kp()
                                self._boost_stuck_t0 = None
                                self._boost_tries = 0
                                self._boost_gave_up = False   # ★ 收敛即解除闩锁
                            elif now > self._boost_deadline:
                                # 超时未收敛: 恢复+计次(连续3次放弃, 防堵转发热)
                                self._boost_kp_on = False
                                self._apply_arm_kp()
                                self._boost_stuck_t0 = None
                                self._boost_tries += 1
                                if self._boost_tries >= self._BOOST_MAX_TRIES:
                                    self._log("[boost] %d次助推未收敛, 放弃"
                                              "(疑似堵转/过流, 请检查电机)" % self._boost_tries)
                                    self._boost_gave_up = True   # ★ 闩锁: 同一卡滞不再重试
                                    self._health_check()   # ★ 助推高kp可能触发过流保护关输出
                            else:
                                # 助推中: 持续刷新; ★ 2026-10-31 J1 kp渐进爬坡
                                # (原理见 _BOOST_KP_MAX 注释)
                                self._apply_arm_kp()
                                _t_boost = now - (self._boost_deadline
                                                  - self._BOOST_TIMEOUT_S)
                                _ramp = min(self._BOOST_KP
                                            + 25.0 * int(_t_boost / 0.4),
                                            self._BOOST_KP_MAX)
                                for j in elig:
                                    ag._mit_kp[j] = _ramp
                                if _ramp > self._BOOST_KP and _ramp != getattr(
                                        self, "_boost_ramp_last", None):
                                    self._boost_ramp_last = _ramp
                                    self._log("[boost] kp爬坡→%g" % _ramp)
                        elif (elig and self._boost_tries < self._BOOST_MAX_TRIES
                              and not self._boost_gave_up):
                            # 目标静止+残差卡在静摩擦带内: 累计卡滞时长
                            # ★ gave_up 闩锁期间不再触发(见 __init__ 注释)
                            if self._boost_stuck_t0 is None:
                                self._boost_stuck_t0 = now
                            elif now - self._boost_stuck_t0 >= self._BOOST_STUCK_S:
                                self._boost_kp_on = True
                                self._boost_deadline = now + self._BOOST_TIMEOUT_S
                                self._boost_ramp_last = self._BOOST_KP
                                self._apply_arm_kp()
                                for j in elig:
                                    ag._mit_kp[j] = self._BOOST_KP
                                self._log("[boost] 关节%s 卡滞(err=%s) kp→%g 冲破静摩擦" % (
                                    elig, ["%.3f" % err[j] for j in elig], self._BOOST_KP))
                        else:
                            self._boost_stuck_t0 = None
                            # ★ 闩锁解除条件: 助推关节真正收敛(err<ERR_LO).
                            #   注意 elig 为空可能只是"残差出带"(收敛或超带),
                            #   只有 <ERR_LO 才算真收敛, 出带上限不算.
                            if self._boost_gave_up:
                                _conv = all(err[j] < self._BOOST_ERR_LO
                                            for j in self._BOOST_ONLY_INDEX
                                            if j < n)
                                if _conv:
                                    self._boost_gave_up = False
                                    self._boost_tries = 0
                                    self._log("[boost] 关节已收敛, 解除放弃闩锁")
                    else:
                        # 重力计算失败: 保守起见本拍不助推
                        if self._boost_kp_on:
                            self._boost_kp_on = False
                            self._apply_arm_kp()
                        self._boost_stuck_t0 = None
        except Exception:
            pass   # 助推逻辑异常绝不阻断控制循环
        # ★ 1Hz 定期电机健康巡检: 兜住非助推原因导致的瞬态失能
        try:
            now = time.monotonic()
            if now - self._hbeat_last >= 1.0:
                self._hbeat_last = now
                self._health_check()
        except Exception:
            pass
        self.ctrl._loop_cb(arm, dt)

    def _connect(self):
        """★ 安全连接(替代直接 ctrl.start()): 镜像 start() 流程但补三道保险:
        ① 逐电机 clear_error 再使能(过流/跟随误差锁死的电机 enable 不报错
           但不响应指令, 表现为"连接了但臂不动")
        ② 软增益覆盖(yaml 硬增益 kp=150 运动会过流锁死/电流冲击):
           常态 kp=25(kd=2), J1 单独 40(扛整臂静摩擦大)
        ③ 使能前 get_state 同步 _q_target=当前实测角(否则臂不在零位会被
           猛拽回零)"""
        # ★ 2026-10-30: 电机失能后用户手动点「启动机械臂」走 connect —— 此时
        #   控制循环仍在跑(失能只停轨迹, 不杀循环), 直接 start_control_loop
        #   会抛 "控制循环已在运行" 导致手动恢复失败. 先停旧循环再重连.
        if getattr(self.rebotarm, "control_loop_active", False):
            self._log("[connect] 检测到控制循环仍在运行, 先停止旧循环")
            try:
                self.rebotarm.stop_control_loop()
            except Exception as e:
                self._log("[connect] stop_control_loop 失败: %s" % e)
        self.rebotarm.connect()
        # ★ CAN 反馈超时(2026-10-19): 关节 2s / 夹爪 0.5s —— motorbridge 默认
        #   30s, 总线打满(ENOBUFS)期间 move_to_traj 内部 get_state 会等关节
        #   电机整整 30s("转移·目标上方卡30s"根因). 正常反馈 <100ms, 2s 充裕;
        #   夹爪会失联(堵转过流)收紧到 0.5s.
        for jc in self.rebotarm._all_joints:
            try:
                ms = 500 if "gripper" in jc.name.lower() else 2000
                self.rebotarm._motor_map[jc.name].set_can_timeout_ms(ms)
            except Exception:
                pass
        for jc in self.rebotarm._all_joints:
            try:
                self.rebotarm._motor_map[jc.name].clear_error()
            except Exception as e:
                self._log("[connect] clear_error %s: %s" % (jc.name, e))
        ag = self.ctrl._arm_group
        if ag is not None:
            self._apply_arm_kp()          # ★ BASE=25, J1=50, J3=40, J4=25
            ag._mit_kd[:] = self._ARM_KD   # 阻尼全局=2.0
            ag._mit_kd[2] = self._ARM_KD_J3   # ★ J3 专属 kd=3(压高kp振荡)
            ag.mode_mit()
            q, _, _ = self.rebotarm.get_state()
            self.ctrl._q_target[:] = q[:self.ctrl._n]
            ag.enable()
        if self.ctrl._has_gripper:
            # ★ 夹爪柔顺增益 kp=10/kd=1(对齐旧服务器, 2026-10-18):
            #   ① UI 匍匐标定的触点判定以 kp=10 设计(触点压力=kp×0.3rad≈3力矩),
            #     yaml 硬增益 kp=50 顶触点=15力矩 -> 持续堵转过流失效
            #     ("标定完夹爪没电"的根因);
            #   ② 大步长目标即使有 ramp, 软增益也杜绝满力矩冲击.
            gp = self.ctrl._gripper_group
            gp._mit_kp[:] = 10.0
            gp._mit_kd[:] = 1.0
            gp.mode_mit()
            # ★ 夹爪目标同步(2026-10-25): _gripper_target 初始 0.0rad(≈5mm 并拢位),
            #   不同步则控制循环启动瞬间 kp=18×大位置误差 ≈ 满力矩猛冲
            #   —— "连接后夹爪快速猛闭"根因(臂的 _q_target 有同步, 夹爪漏了)
            try:
                qg, _, _ = self.rebotarm.get_state()
                self.ctrl._gripper_target = float(qg[self.ctrl._n])
            except Exception as e:
                self._log("[connect] 夹爪目标同步失败(保持现值): %s" % e)
            gp.enable()
        self._boost_reset()   # ★ 助推状态清零(kp可能残留60)
        self._motor_fault = False   # ★ 用户手动启动 → 重新启用失能巡检
        self._motors_enabled = True
        self.rebotarm.start_control_loop(self._loop_cb_boost, rate=250)
        self.ctrl._running = True
        return True

    def _solve_ik(self, x, y, z, rpy=None, normal=None):
        """★ 轻量IK(只解算不生成轨迹): UI 动作线程/画框法向预筛用.
        连续性种子=当前指令目标 _q_target(不做CAN往返, 不占总线);
        FK验收: 位置残差>8mm 判不可达(防 IK 假成功).
        ★ 用独立 _ik_data(见模块头注释)."""
        global _ik_data
        n = self.ctrl._n
        if _ik_data is None:
            _ik_data = _model.createData()
        q_seed = pad_q_for_model(
            _model, np.asarray(self.ctrl._q_target[:n], dtype=float), n)
        if normal is not None:
            roll, pitch, yaw = self.normal_to_rpy(*normal)
        elif rpy is not None:
            roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
        else:
            roll = pitch = yaw = 0.0
        pos = np.array([x, y, z], dtype=float)
        T = pos_rot_to_se3(pos, roll=roll, pitch=pitch, yaw=yaw)
        res = solve_ik(_model, _ik_data, self.ctrl._end_frame_id, T, q_seed,
                       self.ctrl._ik_solver_params, controlled_joints=n)
        # ★ 2026-10-25: 单种子不收敛时随机多种子重试(≤8次) —— 涂抹/粘取2号位
        #   (x=0.2)等远端点从当前姿态出发的单种子CLIK易陷局部极小, 校验误报
        #   "不可达"(而「移动到位置」走的 move_traj 内部是多种子IK能解出,
        #   表现为"能移动到但点动作按钮报不可达"). 只放宽可达性判定,
        #   FK验收(8mm)与运动解耦不变.
        if not res.success:
            lo, hi = _model.lowerPositionLimit, _model.upperPositionLimit
            best = res
            for _ in range(8):
                q_rand = np.array([
                    np.random.uniform(lo[j] if np.isfinite(lo[j]) else -np.pi,
                                      hi[j] if np.isfinite(hi[j]) else np.pi)
                    for j in range(_model.nq)])
                r2 = solve_ik(_model, _ik_data, self.ctrl._end_frame_id, T,
                              q_rand, self.ctrl._ik_solver_params,
                              controlled_joints=n)
                if r2.error < best.error:
                    best = r2
                if best.success:
                    break
            res = best
        q_sol = res.q[:n]
        # FK验收: IK success 可能是假收敛, 用实际末端位置残差把关
        qp = pad_q_for_model(_model, q_sol, n)
        pin.forwardKinematics(_model, _ik_data, qp)
        pin.updateFramePlacements(_model, _ik_data)
        pe = float(np.linalg.norm(
            np.asarray(_ik_data.oMf[self.ctrl._end_frame_id].translation) - pos))
        if res.success or pe < 0.008:
            return True, "ok (pos_err=%.4f)" % pe, [float(v) for v in q_sol]
        return False, "unreachable (pos_err=%.4f)" % pe, None

    def _recover(self):
        """★ 电机错误恢复(UI 看门狗/断线自愈后会发 recover):
        停轨迹 → 逐电机清错误 → 直读固件0x7019同步目标(防使能猛拽)
        → 软增益重切MIT → 使能 → 逐电机验证在线."""
        self._stop_traj()
        self._boost_reset()   # ★ 助推状态清零(恢复流程自己会重写kp=25)
        # ★ 同 _connect: CAN 超时 关节2s/夹爪0.5s
        for jc in self.rebotarm._all_joints:
            try:
                ms = 500 if "gripper" in jc.name.lower() else 2000
                self.rebotarm._motor_map[jc.name].set_can_timeout_ms(ms)
            except Exception:
                pass
        for jc in self.rebotarm._all_joints:
            try:
                self.rebotarm._motor_map[jc.name].clear_error()
            except Exception as e:
                self._log("[recover] clear_error %s: %s" % (jc.name, e))
            time.sleep(0.02)
        n = self.ctrl._n
        # ★ 直读 RobStride 固件寄存器 0x7019(get_state 缓存可能冻结在零值)
        q_live = []
        try:
            for jc in self.rebotarm._all_joints[:n]:
                q_live.append(float(self.rebotarm._motor_map[jc.name]
                                    .robstride_get_param_f32(0x7019, timeout_ms=600)))
        except Exception as e:
            self._log("[recover] live位置读取失败(沿用当前目标): %s" % e)
            q_live = []
        if q_live:
            self.ctrl._q_target[:] = q_live
            self.ctrl._qd_target[:] = 0.0
            self._log("[recover] live位置同步: %s (deg)" % [
                round(float(np.degrees(v)), 1) for v in q_live])
        ag = self.ctrl._arm_group
        if ag is not None:
            self._apply_arm_kp()          # ★ BASE=25, J1=50, J3=40, J4=25
            ag._mit_kd[:] = self._ARM_KD   # 阻尼全局=2.0
            ag._mit_kd[2] = self._ARM_KD_J3   # ★ J3 专属 kd=3(与 connect 一致)
            ag.mode_mit()
        if self.rebotarm.has_gripper:
            # ★ 同 _connect: 夹爪柔顺增益 kp=10/kd=1(触点判定设计前提)
            gp = self.ctrl._gripper_group
            gp._mit_kp[:] = 10.0
            gp._mit_kd[:] = 1.0
            gp.mode_mit()
            # ★ 夹爪目标同步(2026-10-25, 直读0x7019): recover 前夹爪多因堵转
            #   过流失联, get_state 缓存可能冻结; 读失败保持现值(失联电机
            #   本就不响应指令, 不会猛冲)
            try:
                for jc in self.rebotarm._all_joints:
                    if "gripper" in jc.name.lower():
                        self.ctrl._gripper_target = float(
                            self.rebotarm._motor_map[jc.name]
                            .robstride_get_param_f32(0x7019, timeout_ms=600))
                        break
            except Exception as e:
                self._log("[recover] 夹爪目标同步失败(保持现值): %s" % e)
        self.rebotarm.enable_all()
        # 验证离线关节(软件 clear_error 解不掉时需动力电断电 10s)
        offline = []
        for jc in self.rebotarm._all_joints[:n]:
            try:
                self.rebotarm._motor_map[jc.name].robstride_get_param_f32(
                    0x7019, timeout_ms=600)
            except Exception:
                offline.append(jc.name)
        if offline:
            self._log("[recover] ★ 仍离线: %s (软件解不掉, 断电10s后重跑)" % offline)
            return {"ok": False, "err": "仍离线关节: %s (断电10s)" % offline}
        self._motor_fault = False   # ★ 用户手动恢复成功 → 重新启用失能巡检
        self._motors_enabled = True
        self._log("[recover] clear_error + live同步 + 软增益 + mode_mit + enable OK")
        return {"ok": True}


    def normal_to_rpy(self, nx, ny, nz, roll=0.0):
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



    def move_traj(self, x, y, z, normal=None, q6=None, duration=None):
        """服务器端测地线轨迹: RebotArmEndPose.move_to_traj(IK+CLIK跟踪).
        返回成功与否(供 _handle 回包, UI 按此判可达性)."""
        if normal is None:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        else:
            roll, pitch, yaw = self.normal_to_rpy(*normal)
        if q6 is not None:
            roll = float(q6)   # ★ 腕部自转(J6) = 绕接近轴的横滚
        dur = self.duration if duration is None else float(duration)

        # ★ 2026-11-05: 新轨迹启动前停掉进行中的助推(还原kp, 防起步顿挫).
        # 用 _boost_halt 而非 _boost_reset: 保留失败计次/放弃闩锁 — 否则
        # 动作里每段轨迹结束 J1 又卡滞 → 助推重试 → kp=150 电流尖峰
        # 打爆 48V 母线 → 关节静默失能("动作做完就掉线"的根因).
        self._boost_halt()

        ok = bool(self.ctrl.move_to_traj(
                    x=x, y=y, z=z,
                    roll=roll, pitch=pitch, yaw=yaw,
                    duration=dur,
                ))
        if ok:
            # move_to_traj 是异步的(返回时轨迹线程刚启动), 验收放后台线程,
            # 避免阻塞 RPC 响应(UI 靠 traj_active 下降沿判完成)
            self._verify_seq = getattr(self, "_verify_seq", 0) + 1
            threading.Thread(target=self._verify_arrival,
                             args=(self._verify_seq, x, y, z, normal, q6),
                             daemon=True).start()
        return ok

    def _boost_settled(self):
        """★ 2026-11-07 到位验收前的助推状态判据: 所有助推关节要么真收敛
        (err<ERR_LO), 要么已放弃(gave_up), 要么不在静摩擦带(助推帮不上),
        才算"助推已收尾" —— 验收/补偿不会跟助推抢同一关节."""
        try:
            ag = self.ctrl._arm_group
            n = self.ctrl._n
            if ag is None or n <= 0:
                return True
            qt = np.asarray(self.ctrl._q_target[:n], dtype=float)
            qm = np.asarray(ag.get_positions(request_feedback=False),
                            dtype=float)
            if qm.size != n:
                return True
            for j in self._BOOST_ONLY_INDEX:
                if j >= n:
                    continue
                e = abs(float(qt[j]) - float(qm[j]))
                if e < self._BOOST_ERR_LO:
                    continue                     # 该关节已真收敛
                if getattr(self, "_boost_gave_up", False):
                    continue                     # 已放弃, 等也无用
                if self._BOOST_ERR_LO <= e <= self._BOOST_ERR_HI:
                    return False                 # 带内未收敛: 助推还会/正在推
            return True
        except Exception:
            return True

    def _verify_arrival(self, seq, x, y, z, normal=None, q6=None):
        """★ 到位验收(2026-10-31, 定位"目标输入 vs FK显示差很多"):
        等轨迹真正结束+停稳1.2s, 读实测角缓存做 FK, 对比目标点 ——
        打印逐关节残差(deg) + 末端总偏差(mm), 谁拖后腿一眼可见.
        FK 用 compute_fk(自建Data, 与控制循环 _data 无竞争);
        位置读缓存(request_feedback=False), 零 CAN 开销."""
        try:
            t0 = time.time()
            while getattr(self.ctrl, "_moving", False):
                if time.time() - t0 > 90.0:
                    return
                if self._verify_seq != seq:
                    return   # 新轨迹已发起, 本次作废
                time.sleep(0.2)
            # ★ 2026-11-07 等助推收尾再验收: 轨迹点发完(_moving=False)时
            #   J1 静摩擦卡滞的助推往往才刚起步(0.5s 判定 + 3s×3 次爬坡),
            #   原 1.2s 停稳窗内助推还没推完就验收 → 误判 ~17mm 偏差 →
            #   补偿指令叠在助推头上(过冲/IK失败震荡). 先等助推真收敛或放弃.
            time.sleep(0.6)   # 给卡滞判定窗(_BOOST_STUCK_S=0.5s)留时间
            _t0b = time.monotonic()
            while not self._boost_settled():
                if self._verify_seq != seq:
                    return
                if time.monotonic() - _t0b > 13.0:
                    break
                time.sleep(0.2)
            time.sleep(1.2)   # 等关节停稳
            if self._verify_seq != seq:
                return
            n = self.ctrl._n
            ag = self.ctrl._arm_group
            qm = np.asarray(ag.get_positions(request_feedback=False),
                            dtype=float)
            if qm.size != n:
                return
            qt = np.asarray(self.ctrl._q_target[:n], dtype=float)
            deg = ["%.2f" % float(np.degrees(qt[i] - qm[i]))
                   for i in range(n)]
            from reBotArm_control_py.kinematics import compute_fk
            q_pad = pad_q_for_model(self.ctrl._model, qm, n)
            pos, _, _ = compute_fk(self.ctrl._model, q_pad)
            err_mm = float(np.linalg.norm(
                pos - np.array([x, y, z], dtype=float))) * 1000.0
            self._log("[到位验收] 目标(%.4f,%.4f,%.4f) FK实测(%.4f,%.4f,%.4f)"
                      " 末端偏差=%.1fmm | 逐关节残差deg(J1~J%d)=%s"
                      % (x, y, z, pos[0], pos[1], pos[2], err_mm, n, deg))
            # ★ 到位闭环一次性补偿(2026-10-31 方案A): 偏差超阈值且非异常
            #   量级时补一条短修正轨迹; 新指令/失能/急停自动放弃
            if (self._COMP_ERR_THRESH_MM < err_mm <= self._COMP_ERR_MAX_MM
                    and self._verify_seq == seq):
                self._arrival_compensate(seq, x, y, z, normal, q6)
        except Exception as e:
            self._log("[到位验收] 异常(不影响控制): %s" % e)

    def _arrival_compensate(self, seq, x, y, z, normal, q6):
        """★ 到位闭环一次性补偿(2026-10-31 方案A, 用户选定):
        纯位置式MIT存在姿态相关的稳态残差(J3低位实测-1.1°→末端7mm), 固定
        重力前馈系数无法全姿态归零(高位刚好/低位欠补). 停稳后实测末端偏差,
        超阈值就以"目标+实测误差向量"补发一条短轨迹 —— IK以当前实测关节
        为种子, mm级微调不会跳解支.
        与被否决的"目标游走"(旧方案A)本质不同: 那是运动中连续拖动目标,
        关节跟不上导致超调; 这是停稳后一次性离散补刀, 最多 _COMP_MAX_ITER 次.
        时序安全: 下一段 move_traj 会 bump _verify_seq 并取消进行中的补偿
        轨迹 → 循环立即退出让位; UI 每段后 get_state 同步台账, 补偿结果
        自动进入后续段起点, 无误差累积. 涂抹往复等快节奏段补偿常被下一段
        取消属预期(无停稳语义); 架位/目标点/回收位等关键停驻点UI有2s停驻,
        补偿可完整执行."""
        from reBotArm_control_py.kinematics import compute_fk
        try:
            n = self.ctrl._n
            for _it in range(1, self._COMP_MAX_ITER + 1):
                if self._verify_seq != seq or getattr(self, "_motor_fault", False):
                    return
                if not getattr(self, "_motors_enabled", False):
                    return   # 用户失能/急停后绝不动臂
                qm = np.asarray(self.ctrl._arm_group.get_positions(
                    request_feedback=False), dtype=float)
                if qm.size != n:
                    return
                pos, _, _ = compute_fk(
                    self.ctrl._model, pad_q_for_model(self.ctrl._model, qm, n))
                err = np.array([x, y, z], dtype=float) - pos
                err_mm = float(np.linalg.norm(err)) * 1000.0
                if err_mm <= self._COMP_ERR_THRESH_MM:
                    self._log("[到位补偿] 已收敛: 剩余 %.1fmm ≤ %.1fmm"
                              % (err_mm, self._COMP_ERR_THRESH_MM))
                    return
                cx, cy, cz = x + err[0], y + err[1], z + err[2]
                self._log("[到位补偿] 第%d次: 偏差 %.1fmm → 补偿目标 (%.4f, %.4f, %.4f)"
                          % (_it, err_mm, cx, cy, cz))
                # 姿态与原轨迹同源(normal_to_rpy + q6覆盖roll, 同 move_traj)
                if normal is None:
                    roll, pitch, yaw = 0.0, 0.0, 0.0
                else:
                    roll, pitch, yaw = self.normal_to_rpy(*normal)
                if q6 is not None:
                    roll = float(q6)
                if not bool(self.ctrl.move_to_traj(
                        x=cx, y=cy, z=cz,
                        roll=roll, pitch=pitch, yaw=yaw, duration=1.5)):
                    self._log("[到位补偿] 补偿段IK失败, 放弃")
                    return
                t0 = time.time()
                while (time.time() - t0 < 10.0
                       and getattr(self.ctrl, "_moving", False)):
                    if self._verify_seq != seq:
                        return   # 新指令已接管, 补偿让位
                    time.sleep(0.1)
                time.sleep(1.0)   # 停稳再测
            # 次数用尽: 报最终残余(仍大→提示查失能, 不再硬补)
            if self._verify_seq != seq:
                return
            qm = np.asarray(self.ctrl._arm_group.get_positions(
                request_feedback=False), dtype=float)
            if qm.size == n:
                pos, _, _ = compute_fk(
                    self.ctrl._model, pad_q_for_model(self.ctrl._model, qm, n))
                err_mm = float(np.linalg.norm(
                    np.array([x, y, z], dtype=float) - pos)) * 1000.0
                if err_mm > self._COMP_ERR_THRESH_MM:
                    self._log("[到位补偿] %d次后仍偏差 %.1fmm — 疑似静默失能或"
                              "跳解支, 请看逐关节残差/手动确认" % (
                                  self._COMP_MAX_ITER, err_mm))
        except Exception as e:
            self._log("[到位补偿] 异常(不影响控制): %s" % e)




    def _handle(self,cmd) :
        c = cmd.get("cmd")

        x = cmd.get("x")
        y = cmd.get("y")
        z = cmd.get("z")

        normal = cmd.get("normal")

        _lock.acquire()
        try:
            if c == "connect":
                ok = self._connect()
                return {"ok": bool(ok), "msg": "connected" if ok else "connect failed",
                        "n_arm": self.ctrl._n,
                        "has_gripper": self.rebotarm.has_gripper}

            # ★ 未连接守卫: ctrl 对象在 __init__ 即创建(永不为None),
            #   真正的判据是运行标志 _running
            if not getattr(self.ctrl, "_running", False):
                return {"ok": False, "err": "not connected"}

            if c == "enable":
                # ★ 真使能(原来是空操作): 先停轨迹+同步目标为当前实测角,
                #   否则使能瞬间会被旧目标猛拽
                self._stop_traj()
                q, _, _ = self.rebotarm.get_state()
                self.ctrl._q_target[:] = q[:self.ctrl._n]
                # ★ 夹爪目标同步(2026-10-25): disable/estop 期间夹爪被手掰动后
                #   重新使能, 旧目标大力矩猛拽(与臂同病, 一并修)
                if self.rebotarm.has_gripper and len(q) > self.ctrl._n:
                    self.ctrl._gripper_target = float(q[self.ctrl._n])
                self.rebotarm.enable_all()
                self._motor_fault = False   # ★ 用户手动使能 → 重新启用失能巡检
                self._motors_enabled = True
                self._log("[enable] enable_all done"); return {"ok": True}

            if c == "disable":
                self._stop_traj(); self._boost_reset()
                self._motors_enabled = False   # ★ 主动去使能: 臂下垂不算失能
                self.rebotarm.disable_all(); return {"ok": True}
            if c == "estop":
                self._stop_traj(); self._boost_reset()
                self._motors_enabled = False   # ★ 急停: 臂下垂不算失能
                self.rebotarm.estop(); return {"ok": True}
            if c == "recover":
                return self._recover()
            if c == "set_joints":
                # ★ 流式关节目标(UI 30Hz: J6微调/回零/预设/扫描/动作线程):
                #   先取消服务器端轨迹(流式优先), 再写指令目标(_loop_cb 250Hz
                #   以 MIT 跟踪). 短列表只覆盖前若干关节, 不会误命令其余到0.
                j = list(cmd["joints"])
                n = self.ctrl._n
                self._stop_traj()
                self.ctrl._q_target[:min(len(j), n)] = j[:n]
                return {"ok": True}
            if c == "get_state":
                # ★ 2026-10-19: request_feedback=False 读缓存 —— 250Hz MIT
                #   指令的电机回帧自动刷新缓存(数据新鲜), 零CAN往返.
                #   原直读(逐电机反馈请求)在 UI 0.1s 轮询下产生大量总线
                #   往返, 与 250Hz 发送叠加 -> 往复涂抹后 ENOBUFS 打满总线
                #   -> move_to_traj 内部 get_state 等关节电机默认 30s 超时
                #   -> "转移·目标上方卡30s"的真正链条. can_ping 仍保留
                #   CAN 往返探测真死.
                q, _, _ = self.rebotarm.get_state(request_feedback=False)
                joints = [float(v) for v in q[:self.ctrl._n]]
                g = float(q[self.ctrl._n]) if self.rebotarm.has_gripper and len(q) > self.ctrl._n else 0.0
                # traj_active: 服务器端轨迹是否仍在执行(UI 阻塞等待的完成判据)
                return {"ok": True, "joints": joints, "gripper": g,
                        "traj_active": bool(self.ctrl._moving)}
            if c == "solve_ik":
                ok, msg, j = self._solve_ik(x, y, z, rpy=cmd.get("rpy"), normal=normal)
                return {"ok": ok, "err": msg, "joints": j}
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
                # ★ 2026-11-05 改 request_feedback=False — 控制循环 250Hz
                #   MIT 指令已自动刷新电机缓存，每次 get_pose 再发 7 帧 CAN
                #   请求既浪费带宽又放大 USB 拥塞风险；热路径抢锁窗口也减小
                q, _, _ = self.rebotarm.get_state(request_feedback=False)
                q_pad = pad_q_for_model(_model, q[:self.ctrl._n], self.ctrl._n)
                pos, rot, _ = compute_fk(_model, q_pad)
                rpy = pin.rpy.matrixToRpy(np.array(rot)).tolist()
                return {"ok": True, "pos": [float(v) for v in pos],
                        "rpy": [float(v) for v in rpy],
                        "joints": [float(v) for v in q[:self.ctrl._n]]}


            if c == "move_traj":
                # ★ 服务器端轨迹: IK+测地线+CLIK 全在 move_to_traj 内完成,
                #   成败回传(必须 return —— 原版穿透到 unknown cmd).
                # ★ 2026-10-18: duration 尊重 cmd 传入; 未传时按距离自适应.
                # ★ 重力前馈保持常开(用户确认: 关FF期间臂下垂不对). 30s卡/ENOBUFS
                #   的真根因已定位为 motorbridge CAN 反馈超时(set_can_timeout_ms
                #   500ms 修复), 与 Data 竞争无关.
                dur = cmd.get("duration")
                if dur is None:
                    dur = self._auto_duration(x, y, z)
                ok = self.move_traj(x, y, z, normal=normal,
                                    q6=cmd.get("q6"),
                                    duration=float(dur))
                return {"ok": ok}

            if c == "set_gripper":
                # ★ 2026-10-18: 改斜坡渐进(_gripper_ramp) —— 直写大步长阶跃
                #   会让 MIT kp=50 的夹爪瞬间满力矩过流锁死("点设定不动"根因)
                self._gripper_ramp(float(cmd["pos"]))
                return {"ok": True}

            if c == "home":
                self._stop_traj(); self.ctrl.safe_home(); return {"ok": True}
            if c == "shutdown":
                # ★ 2026-08-31: 关UI时直接disable(不回零) — 原版end()里safe_home
                #   先回零再断开(10-20秒), UI等30s超时或进程被强杀时电机保持
                #   使能态 → PCAN灯常亮. 现在改立即停控制循环+disable+断开, 2秒内完成.
                try:
                    self._stop_traj()
                    self.rebotarm.stop_control_loop()
                    self.rebotarm.disable_all()
                    self._log("[shutdown] 控制循环已停+全部电机已去使能")
                except Exception as e:
                    self._log("[shutdown] disable异常(仍继续断开): %s" % e)
                try:
                    self.rebotarm.disconnect()   # 内部也做 disable_all
                    self._log("[shutdown] CAN断开完成")
                except Exception as e:
                    self._log("[shutdown] disconnect异常: %s" % e)
                return {"ok": True}
            return {"ok": False, "err": "unknown cmd: %s" % c}

        except Exception as e:
            self._log("EXC:%s\n%s" % (e, traceback.format_exc()))
            # ★ 必须回错误包: 原版只log不return → main 打印 "null" →
            #   UI 端跳过非dict行一直等到超时
            return {"ok": False, "err": str(e)}
        finally:
            _lock.release()



def main() -> None:
    _arm = rebot_control()
    _arm._log("rebot_arm_server ready (rebot_pin py3.11) pid=%d" % os.getpid())
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
            # 控制循环死亡检测: ctrl 未运行(内部异常) = "连接了但发不出指令"
            # (灯常亮假象), 立即报错退出让 UI 触发重启自愈, 不再静默装死
            _c = getattr(_arm, "ctrl", None)
            if _c is not None and not getattr(_c, "_running", False):
                _arm._log("[FATAL] 控制循环已死亡(内部异常), arm_server 退出待UI重启")
                break
            continue
        # ★ rx 打点(2026-10-18 协议断点定位): 与 UI 端 [tx] 时间戳配对,
        #   "UI超时但后端没收到"还是"后端回了UI没收到"一眼定位
        _arm._log("[rx] %s t=%.3f" % (cmd.get("cmd"), time.monotonic()))
        _t0 = time.monotonic()
        resp = _arm._handle(cmd)
        _dt = time.monotonic() - _t0
        if _dt > 2.0:
            # ★ 慢指令诊断打点(2026-10-18): 涂抹步骤6"响应超时"事故定位用
            #   —— 一眼看出是哪条 cmd 在 _handle 里卡了多久
            _arm._log("[slow] cmd=%s 处理耗时 %.1fs" % (cmd.get("cmd"), _dt))
        # ★ 2026-11-07 序列号回显: UI 按序匹配响应, 某命令超时后其滞留
        #   回包不再被后续命令误读(响应错位级联 = "FK 显示未连接但后端
        #   健康"的根因). set_joints 不回包, 无需 seq.
        if resp is None:
            resp = {"ok": False, "err": "null resp"}
        resp["seq"] = cmd.get("seq")
        print(json.dumps(resp), flush=True)
        if cmd.get("cmd") == "shutdown":
            break
    # ★ stdin EOF (UI 崩溃/被杀): 不让电机瞬间掉电失控 ——
    #   保持当前目标发 2 秒 MIT 保持指令, 然后安全断开(断开内部会 disable).
    #   给用户 2 秒窗口期反应 + 电机受控泄力而非猛掉.
    _arm._log("[EOF] stdin 关闭(UI退出?), 进入 2 秒保持后安全退出")
    try:
        import threading as _th
        _t = _th.Timer(2.0, _arm.rebotarm.disconnect)
        _t.daemon = True
        _t.start()
        _t.join()
    except Exception as e:
        _arm._log("[EOF] 安全退出异常: %s" % e)
    _arm._log("rebot_arm_server exit")


if __name__ == "__main__":
    main()

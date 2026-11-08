#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
handeye_sample.py —— 眼在手上(eye-in-hand)手眼标定【终端采样工具】

作用
----
在终端里按回车采集一组样本, 自动把两路数据【同时刻】配对落盘:
  · 标定板位姿: /aruco_single/pose  (ChArUco 板在【相机系】下的位姿)
  · 机械臂位姿: rebot_arm_control.py 的 get_pose (末端在【基座系】的 pos/rpy)
采集完用 cv2.calibrateHandEye 解算 T_gripper<-camera.

用法
----
    # 1) 先确认标定模式已跑起来(UI 里勾「标定模式」+「启动相机」)
    # 2) 另开终端:
    source /opt/ros/noetic/setup.bash
    source ~/forensic_ws/devel/setup.bash
    ~/anaconda3/envs/centernet_4060/bin/python handeye_sample.py

操作
----
    回车 : 采集一组(要求板子当前能被检出, 且机械臂静止)
    d    : 删除上一组
    l    : 列出已采集样本
    s    : 保存并解算 -> handeye_result_YYYYmmdd_HHMMSS.txt
    q    : 退出

采样要点(决定精度, 务必照做)
--------------------------
  1. 板子固定不动, 只动机械臂(这是 eye-in-hand 的标准做法);
  2. 至少 12~15 组, 姿态要【多样】: 绕不同轴倾斜、不同距离、板子出现在
     画面不同区域 —— 姿态越单一, 标定越差(纯平移几乎解不出旋转);
  3. 每组之间末端姿态变化尽量大(>15°), 不要只在原地小幅移动;
  4. 每组采样时机械臂必须【完全静止】(动着采会引入几十 mm 误差).

★ 与 perception_calib.py 的关系
  perception_calib.py 里现有的 REBOT_HAND_EYE_T_GRIPPER_CAM 是用【Gemini
  相机】标出来的. 你现在换成 Deeyea 相机, 相机相对末端的位置/朝向变了,
  该矩阵【必须重新标定】, 旧值不能沿用.
"""
import os
import sys
import json
import math
import time
import subprocess
import threading

import numpy as np
import cv2

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# ── 机械臂后端(与 UI 同一套, 保证 FK 一致) ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ARM_SERVER = os.path.join(_HERE, 'rebot_arm_control.py')
REBOT_PY = next(
    (p for p in [
        os.path.expanduser('~/anaconda3/envs/rebot_pin/bin/python'),
        os.path.expanduser('~/anaconda3/envs/rebot/bin/python'),
        os.path.expanduser('~/下载/enter/envs/rebot/bin/python'),
    ] if os.path.exists(p)),
    os.path.expanduser('~/anaconda3/envs/rebot_pin/bin/python'))


# ══════════════════════════════════════════════════════════════════════════
# 机械臂 FK —— 首选方案: 从 /joint_states 用 pinocchio 算(不抢 CAN 总线)
# ══════════════════════════════════════════════════════════════════════════
class JointStateFK:
    """★ 2026-10-28 新增: 直接订阅 UI 已在发布的 /joint_states, 用
    reBotArm_control_py 的 compute_fk 算末端位姿.

    ★★ 为什么需要它: 原方案是另起一个 rebot_arm_control.py 子进程问 FK,
      但 UI 里【已经有一个后端占着 CAN 总线】—— 第二个实例必然连不上,
      报 "not connected". 而 UI 每 30ms 就把当前关节角发到 /joint_states,
      直接订阅这个话题再算 FK, 既不抢总线, 又与 UI 的位姿完全同源.

    依赖 pinocchio(rebot_pin 环境). 若导入失败会自动降级为 None,
    由调用方回落到子进程方案."""

    def __init__(self):
        self.model = None
        self.joints = None
        self.ts = 0.0
        self._ok = False
        self._fkproc = None
        self._fk_lock = threading.Lock()

    def start(self):
        # ★ 必须在【干净环境】下导入: LD_LIBRARY_PATH 里有 /opt/ros/* 时,
        #   pinocchio 会加载到 ROS 自带的旧 boost_python, 报
        #   "undefined symbol: _Py_fopen". 这里临时剔除再恢复.
        _saved = os.environ.get('LD_LIBRARY_PATH')
        try:
            _kept = [p for p in (_saved or '').split(':')
                     if p and not p.startswith('/opt/ros/')]
            if _kept:
                os.environ['LD_LIBRARY_PATH'] = ':'.join(_kept)
            else:
                os.environ.pop('LD_LIBRARY_PATH', None)
            # ★★ 关键: pinocchio 只装在 rebot_pin 环境, 而本脚本跑在
            #   centernet_4060(有 rospy/cv2). 两个环境互斥, 所以把 FK 拆成
            #   独立子进程(fk_helper.py, 用 rebot_pin 跑) —— 纯数学计算,
            #   完全不碰 CAN 总线, 可以任意多开(不会和 UI 抢总线).
            _helper = os.path.join(_HERE, 'fk_helper.py')
            if not os.path.exists(_helper):
                raise RuntimeError('找不到 fk_helper.py')
            self._fkproc = subprocess.Popen(
                [REBOT_PY, _helper],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
            # 等就绪信号
            _line = self._fkproc.stdout.readline()
            _info = json.loads(_line) if _line.strip().startswith('{') else {}
            if not _info.get('ok'):
                raise RuntimeError('fk_helper 启动失败: %s' % _info.get('err'))
            self._fk_info = _info
            self._compute_fk = None
            self.model = True      # 标记已就绪
        except Exception as e:
            print("[提示] /joint_states+FK 方案不可用(%s), 将回落到子进程查询" % str(e)[:80])
            return False
        finally:
            if _saved is None:
                os.environ.pop('LD_LIBRARY_PATH', None)
            else:
                os.environ['LD_LIBRARY_PATH'] = _saved
        if self.model is None:
            return False
        rospy.Subscriber('/joint_states', JointState, self._cb, queue_size=1)
        self._ok = True
        return True

    def _fk_impl(self, model, q, frame_name=None):
        """pinocchio 正运动学: 返回 (position, rotation, homogeneous).
        ★ 与 rebot_arm_control.py 的 compute_fk 等价(同一 URDF/同一末端帧),
          只是不依赖 motorbridge, 可独立运行."""
        import pinocchio as pin
        data = model.createData()
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.shape[0] != model.nq:
            # 模型 nq 与传入不等时: 补齐(多)或截断(少)
            if q.shape[0] < model.nq:
                q = np.concatenate([q, np.zeros(model.nq - q.shape[0])])
            else:
                q = q[:model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name or self._end_frame)
        if fid >= len(data.oMf):
            fid = model.nframes - 1
        M = data.oMf[fid]
        pos = np.asarray(M.translation, dtype=np.float64).copy()
        rot = np.asarray(M.rotation, dtype=np.float64).copy()
        H = np.eye(4)
        H[:3, :3] = rot
        H[:3, 3] = pos
        return pos, rot, H

    def _cb(self, msg):
        try:
            d = dict(zip(msg.name, msg.position))
            q = [float(d[n]) for n in
                 ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')]
            self.joints = q
            self.ts = time.time()
        except Exception:
            pass

    def fresh(self, max_age=1.0):
        if self.joints is None:
            return None
        if time.time() - self.ts > max_age:
            return None
        return self.joints

    def get_pose(self):
        """返回 {'ok':True,'pos':[x,y,z],'rpy':[r,p,y],'joints':[...]} 或 {'ok':False,...}"""
        if not self._ok:
            return {"ok": False, "err": "FK 不可用"}
        q = self.fresh()
        if q is None:
            return {"ok": False, "err": "等待 /joint_states(UI 是否在运行?)"}
        try:
            with self._fk_lock:
                self._fkproc.stdin.write(
                    json.dumps({"cmd": "fk", "q": [float(v) for v in q]}) + "\n")
                self._fkproc.stdin.flush()
                line = self._fkproc.stdout.readline()
            r = json.loads(line) if line.strip().startswith('{') else {}
            if not r.get('ok'):
                return {"ok": False, "err": r.get('err', 'FK 无响应')}
            return {"ok": True, "pos": r["pos"], "rpy": r["rpy"],
                    "joints": [float(v) for v in q]}
        except Exception as e:
            return {"ok": False, "err": "FK 计算失败: %s" % e}

    def check_static(self, dur=1.5, tol_deg=0.15):
        """★ 2026-10-30 静止校验: 见模块级 _static_check 的说明."""
        return _static_check(lambda: self.fresh(max_age=0.5), dur, tol_deg)

    def stop(self):
        try:
            if self._fkproc is not None and self._fkproc.poll() is None:
                self._fkproc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._fkproc.stdin.flush()
                self._fkproc.terminate()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# 静止校验 (2026-10-30)
# ══════════════════════════════════════════════════════════════════════════
def _static_check(getter, dur=1.5, tol_deg=0.15):
    """判定机械臂是否真的【停稳】, 返回 (ok, 最大变化角(deg) 或 None).

    ★★ 为什么必须查: UI 发布的 /joint_states 在【运动中】是 30Hz 的
      【指令插值角】(rebot_controller_forensic_auto.py update_all 里
      interpolate(...)), 只有静止后后台线程才用 get_state() 把它改写成
      电机【实测】角. 所以臂还在动、或刚动完还没到一个刷新周期时采到的
      关节角是"理论值", 用它算 FK 与相机实测位姿会错开 —— 这正是
      44.txt 里臂只动 12mm、板位姿却变 17~63mm 的来源.

    做法: 连续 dur 秒反复读同一路关节角, 取各关节 (最大值-最小值) 的
    最大值; 小于 tol_deg 才算停稳. 窗口 1.5s 覆盖 ≥3 个刷新周期,
    确保读到的已是实测角(指令→实测的跳变若发生在窗口内一定会被检出).
    """
    t_end = time.time() + dur
    lo = hi = None
    while time.time() < t_end:
        q = getter()
        if q is not None and len(q) >= 6:
            q = np.asarray(q, dtype=float)
            if lo is None:
                lo, hi = q.copy(), q.copy()
            else:
                lo = np.minimum(lo, q)
                hi = np.maximum(hi, q)
        time.sleep(0.05)
    if lo is None:
        return False, None
    dev = float(np.degrees(np.max(hi - lo)))
    return dev <= tol_deg, dev


# ══════════════════════════════════════════════════════════════════════════
# 机械臂 FK 查询(复用 UI 同款后端)
# ══════════════════════════════════════════════════════════════════════════
class ArmFK:
    """通过 JSON-line 管道问 rebot_arm_control.py 要当前末端位姿.

    ★ 注意: 机械臂位姿【不是】ROS 话题, 而是子进程管道 —— 所以终端里
      `rostopic echo` 拿不到, 必须走这个类."""

    def __init__(self):
        self.proc = None
        self._lock = threading.Lock()
        # ★★ 2026-10-28 关键: 必须从 LD_LIBRARY_PATH 剔除 /opt/ros/* 路径,
        #   否则 pinocchio(rebot_pin 环境)会加载到 ROS 自带的旧 boost_python,
        #   报 "undefined symbol: _Py_fopen" 直接起不来.
        #   与 UI 里 _armserver_env() 的逻辑完全一致(那边有同样注释).
        env = dict(os.environ)
        _kept = [p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                 if p and not p.startswith('/opt/ros/')]
        if _kept:
            env['LD_LIBRARY_PATH'] = ':'.join(_kept)
        else:
            env.pop('LD_LIBRARY_PATH', None)
        env.setdefault('PYTHONUNBUFFERED', '1')
        self._env = env

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        if not os.path.exists(ARM_SERVER):
            print("[错误] 找不到后端: %s" % ARM_SERVER)
            return False
        try:
            # ★ stderr 落盘位置: 优先用户主目录, 不可写时退到 /tmp
            #   (主目录只读的受限环境也能跑起来)
            _errpath = None
            for _cand in (os.path.expanduser('~/handeye_arm.err'),
                          '/tmp/handeye_arm.err'):
                try:
                    _errpath = open(_cand, 'a')
                    break
                except Exception:
                    continue
            self.proc = subprocess.Popen(
                [REBOT_PY, ARM_SERVER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=_errpath if _errpath is not None else subprocess.DEVNULL,
                env=getattr(self, '_env', None),
                text=True, bufsize=1)
        except Exception as e:
            print("[错误] 启动机械臂后端失败: %s" % e)
            return False
        # 等后端就绪
        time.sleep(2.0)
        return self.proc.poll() is None

    def _send(self, cmd, timeout=5.0):
        if self.proc is None or self.proc.poll() is not None:
            if not self.start():
                return {"ok": False, "err": "后端未启动"}
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(cmd) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                return {"ok": False, "err": "写入失败: %s" % e}
            t0 = time.time()
            while time.time() - t0 < timeout:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if 'ok' in r:
                    return r
            return {"ok": False, "err": "超时"}

    def get_pose(self):
        return self._send({"cmd": "get_pose"})

    def check_static(self, dur=1.5, tol_deg=0.15):
        """★ 2026-10-30 静止校验: 见模块级 _static_check 的说明."""
        def _g():
            r = self.get_pose()
            return r.get('joints') if r.get('ok') else None
        return _static_check(_g, dur, tol_deg)

    def stop(self):
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# 板位姿订阅
# ══════════════════════════════════════════════════════════════════════════
class BoardPose:
    def __init__(self):
        self.pose = None
        self.ts = 0.0
        rospy.Subscriber('/aruco_single/pose', PoseStamped, self._cb, queue_size=1)

    def _cb(self, msg):
        self.pose = msg
        self.ts = time.time()

    def fresh(self, max_age=0.6):
        """返回新鲜的板位姿(太旧返回 None) —— 采样必须要求板子当前可见."""
        if self.pose is None:
            return None
        if time.time() - self.ts > max_age:
            return None
        return self.pose


# ══════════════════════════════════════════════════════════════════════════
# 位姿工具
# ══════════════════════════════════════════════════════════════════════════
def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def rpy_to_R(r, p, y):
    """★ 与 rebot_arm_control.py 的 pinocchio 约定一致: R = Rz(y)·Ry(p)·Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    return Rz @ Ry @ Rx


def pose_to_T(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, float)
    return T


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def _default_outpath():
    """结果文件路径: 优先主目录, 不可写则用 /tmp."""
    name = 'handeye_result_%s.txt' % time.strftime('%Y%m%d_%H%M%S')
    for d in (os.path.expanduser('~'), '/tmp'):
        try:
            if os.access(d, os.W_OK):
                return os.path.join(d, name)
        except Exception:
            continue
    return os.path.join('/tmp', name)


class Sample(tuple):
    """一条手眼标定样本.

    ★ 元组部分(前4项)是 cv2.calibrateHandEye 直接要的:
        (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
      所以既有的解包写法 `for (Rg,tg,Rt,tt) in samples` 完全不受影响.
    ★ 其余字段只作留档(与历史 11.txt 对齐), 【不参与解算】:
        joints      : 6 个关节角(rad) —— 可用同一 URDF 复算 FK 交叉验证
        normal      : 末端工具轴法向(由 rpy 派生) —— 历史文件的 "法向n(实际)"
        rpy         : 末端 rpy(rad)
        ts          : 采集时刻(Unix 秒)
        board_quat  : 板位姿四元数 (x,y,z,w)
    """

    # ★ 不要加 __slots__=() —— 那会禁掉实例 __dict__, 额外字段就挂不上去
    def __new__(cls, Rg, tg, Rt, tt, joints=None, normal=None, rpy=None,
                ts=None, board_quat=None):
        self = super().__new__(cls, (Rg, tg, Rt, tt))
        self.joints = joints
        self.normal = normal
        self.rpy = rpy
        self.ts = ts
        self.board_quat = board_quat
        return self

    # 便捷访问
    @property
    def Rg(self): return self[0]
    @property
    def tg(self): return self[1]
    @property
    def Rt(self): return self[2]
    @property
    def tt(self): return self[3]


def _rot_angle_deg(Ra, Rb):
    """两个旋转矩阵之间的夹角(度)."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _nearest_pose_deg(R_new, R_list):
    """新姿态与已有样本中【最接近】者的夹角(度); 无样本返回 None."""
    if not R_list:
        return None
    return min(_rot_angle_deg(R_new, R) for R in R_list)


def _board_base_estimate(samples):
    """★ 用已采集样本【估计板在基座系的位置】, 用于检测板子是否在采样中移动.

    原理: 眼在手上标定假设板子固定, 即
        T_base<-board = T_base<-gripper(i) · X · T_camera<-board(i)
    对每个样本 i, 左边应该是【同一个常量】. 但 X 还没解出来.
    这里先用当前样本解一个粗 X(cv2.calibrateHandEye), 再回代每组求
    T_base<-board, 看这些点是否重合.

    返回 (positions Nx3, spread_m, worst_dev_m) 或 None(样本不足).
    ★ 样本 <4 组时 calibrateHandEye 不稳定, 返回 None 不做判断."""
    n = len(samples)
    if n < 4:
        return None
    try:
        Rg = [s[0] for s in samples]
        tg = [s[1] for s in samples]
        Rt = [s[2] for s in samples]
        tt = [s[3] for s in samples]
        R, t = cv2.calibrateHandEye(
            Rg, [np.asarray(v, float).reshape(3, 1) for v in tg],
            Rt, [np.asarray(v, float).reshape(3, 1) for v in tt],
            method=cv2.CALIB_HAND_EYE_PARK)
        X = pose_to_T(np.asarray(t, float).ravel(), R)
        pos = []
        for i in range(n):
            T_bg = pose_to_T(np.asarray(tg[i], float), Rg[i])
            T_ct = pose_to_T(np.asarray(tt[i], float), Rt[i])
            pos.append((T_bg @ X @ T_ct)[:3, 3])
        pos = np.array(pos)
        ctr = pos.mean(axis=0)
        dev = np.linalg.norm(pos - ctr, axis=1)
        spread = float(np.linalg.norm(pos.std(axis=0)))
        return pos, spread, float(dev.max())
    except Exception:
        return None


def _board_stability_report(samples, thresh_mm=15.0):
    """★ 板位姿稳定性检查: 返回 (是否通过, 提示文本).

    判据: 反算的板心在基座系散布 spread(3D标准差).
      spread <  thresh*0.5   -> 稳定
      spread <  thresh       -> 临界, 提醒注意
      spread >= thresh       -> 板子在动, 警告(这批数据不可用)

    ★ 实测参考(2026-10-28): 板子被手持移动的那批数据 spread=40mm;
      正确固定时应在 5mm 以内.
    ★★ 2026-10-28 样本数门槛 4->6: 检查依赖先用当前样本解一个粗略手眼矩阵 X
      再回代. 4~5 组时 X 本身病态(即使姿态覆盖足够), 回代散布虚高,
      会把"板子没动"误报成"移动了"(实测 4 组/31°覆盖散布 57mm, 实为
      X 估不准). 6 组起判定才有效; 且散布大时列出逐组偏离, 便于区分
      "整体散"(X/板动)与"个别坏样本"(按 d 删该组即可)."""
    n = len(samples)
    if n < 6:
        if n >= 4:
            return True, ("   [稳定性检查待命: %d 组<6, 粗略X病态, 散布数字暂不可信,"
                          " 采到 6 组后开始判定]" % n)
        return True, ""
    r = _board_base_estimate(samples)
    if r is None:
        return True, ""
    pos, spread, worst = r
    spread_mm = spread * 1000.0
    worst_mm = worst * 1000.0
    ctr = pos.mean(axis=0)
    dev = np.linalg.norm(pos - ctr, axis=1) * 1000.0
    # 逐组偏离明细(散布大时显示, 快速定位坏样本)
    detail = ""
    if spread_mm >= thresh_mm * 0.5:
        _worst_i = int(np.argmax(dev))
        detail = "\n   逐组偏离(mm): [%s]  ← 最差: 第%02d组" % (
            ", ".join("%.0f" % v for v in dev), _worst_i + 1)
    if spread_mm < thresh_mm * 0.5:
        return True, ("   板位姿稳定: 基座系散布 %.1fmm (最大偏离 %.1fmm)%s"
                      % (spread_mm, worst_mm, detail))
    if spread_mm < thresh_mm:
        return True, ("   板位姿临界: 散布 %.1fmm — 板子可能轻微移动, 请检查是否固定牢%s"
                      % (spread_mm, detail))
    return False, (
        "   ⚠⚠ 板位姿不稳定: 基座系散布 %.1fmm (最大偏离 %.1fmm) — 板子在采样中移动了!\n"
        "       眼在手上标定要求板子绝对固定. 请把板子粘牢/夹紧后【重新采样】,\n"
        "       这批数据继续采下去也是无效的(建议按 d 逐组删除或重来).%s"
        % (spread_mm, worst_mm, detail))


def _progress_hint(samples):
    """采集过程中的实时质量提示: 告诉用户还差什么."""
    n = len(samples)
    if n < 3:
        return "进度 %d 组 — 还早, 至少需要 12 组(数学下限 3 组)" % n
    Rs = [s[0] for s in samples]
    # 姿态覆盖度: 所有样本两两夹角的最大值
    max_sep = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            max_sep = max(max_sep, _rot_angle_deg(Rs[i], Rs[j]))
    if n < 12:
        tail = "还需约 %d 组" % (12 - n)
    else:
        tail = "数量已够, 可继续加密或直接按 s 解算"
    quality = ("姿态覆盖不足(最大夹角 %.0f°, 建议 >30°)" % max_sep
               if max_sep < 30 else "姿态覆盖良好(最大夹角 %.0f°)" % max_sep)
    return "进度 %d 组 — %s; %s" % (n, tail, quality)


def solve_and_save(samples, outpath):
    """samples: [(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam), ...]
    ★ cv2.calibrateHandEye 的约定(eye-in-hand):
        R_gripper2base/t_gripper2base : 基座<-末端   (机械臂 FK 给的位姿)
        R_target2cam/t_target2cam     : 相机<-标定板 (/aruco_single/pose)
      返回的 X = T_cam2gripper (相机在末端坐标系下的位姿)."""
    Rg = [s[0] for s in samples]
    tg = [s[1] for s in samples]
    Rt = [s[2] for s in samples]
    tt = [s[3] for s in samples]

    results = {}
    for name, method in [
        ('TSAI', cv2.CALIB_HAND_EYE_TSAI),
        ('PARK', cv2.CALIB_HAND_EYE_PARK),
        ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD),
        ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]:
        try:
            R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=method)
            results[name] = (R, t)
        except Exception as e:
            print("  [%s] 解算失败: %s" % (name, e))

    if not results:
        print("[错误] 全部方法解算失败")
        return None

    # 一致性检查: 各方法结果应互相接近
    print("\n各方法结果对比(平移 t, 单位 m):")
    names = list(results.keys())
    for n in names:
        R, t = results[n]
        print("  %-11s t = [% .4f, % .4f, % .4f]  |t| = %.4f"
              % (n, t[0, 0], t[1, 0], t[2, 0], np.linalg.norm(t)))
    base = results[names[0]]
    for n in names[1:]:
        d = np.linalg.norm(results[n][1] - base[1])
        print("  %s 与 %s 平移差: %.2f mm %s"
              % (n, names[0], d * 1000, "✔" if d < 0.01 else "⚠ 偏大"))

    # 回代残差: 板心在基座系的离散度(越小越准)
    print("\n回代残差(板心在基座系的位置离散度):")
    best_name, best_std = None, 1e9
    for n, (R, t) in results.items():
        X = pose_to_T(t.ravel(), R)     # T_cam2gripper
        pts = []
        for Rg_i, tg_i, Rt_i, tt_i in samples:
            T_bg = pose_to_T(tg_i.ravel(), Rg_i)      # 基座<-末端
            T_ct = pose_to_T(tt_i.ravel(), Rt_i)      # 相机<-板
            pts.append((T_bg @ X @ T_ct)[:3, 3])
        pts = np.array(pts)
        std = pts.std(axis=0)
        rms = float(np.linalg.norm(std))
        print("  %-11s std = (%.1f, %.1f, %.1f) mm  3D=%.1f mm"
              % (n, std[0] * 1000, std[1] * 1000, std[2] * 1000, rms * 1000))
        if rms < best_std:
            best_std, best_name = rms, n
    print("\n★ 推荐采用: %s (残差最小 %.1f mm)" % (best_name, best_std * 1000))

    R, t = results[best_name]
    T = pose_to_T(t.ravel(), R)

    with open(outpath, 'w') as f:
        def w(s=''):
            f.write(s + '\n')
            print(s)
        w("=" * 70)
        w("reBot 眼在手上手眼标定结果 (handeye_sample.py)")
        w("时间: %s" % time.strftime('%Y-%m-%d %H:%M:%S'))
        w("样本数: %d" % len(samples))
        w("采用方法: %s (回代残差 %.2f mm)" % (best_name, best_std * 1000))
        w("=" * 70)
        w()
        w("★ T_cam2gripper = 相机在【末端(gripper_end)坐标系】下的位姿")
        w("   用法: p_gripper = T_cam2gripper · p_cam")
        w("   反变换 T_gripper2cam = inv(T_cam2gripper)")
        w()
        w("T_cam2gripper (4x4):")
        for row in T:
            w("  [% .8f % .8f % .8f % .8f]" % tuple(row))
        w()
        w("平移 t (m): [% .6f, % .6f, % .6f]   |t| = %.6f m"
          % (t[0, 0], t[1, 0], t[2, 0], np.linalg.norm(t)))
        rvec, _ = cv2.Rodrigues(R)
        w("旋转向量(轴角): [% .6f, % .6f, % .6f]" % tuple(rvec.ravel()))
        w()
        w("── 各方法交叉验证 ──")
        for n in names:
            Rn, tn = results[n]
            w("  %-11s t = [% .6f, % .6f, % .6f]" % (n, tn[0, 0], tn[1, 0], tn[2, 0]))
        w()
        w("── 原始样本(供复算; 格式对齐历史 11.txt) ──")
        for i, smp in enumerate(samples):
            Rg_i, tg_i, Rt_i, tt_i = smp[0], smp[1], smp[2], smp[3]
            w("  【样本%02d】" % (i + 1))
            # 关节角(可选): 可用同一 URDF 复算 FK 交叉验证
            _j = getattr(smp, 'joints', None)
            if _j:
                w("    关节角度(rad): [%s]"
                  % ", ".join("%.4f" % v for v in _j))
            w("    实际末端位置(FK): X=%.4f Y=%.4f Z=%.4f"
              % (tg_i[0], tg_i[1], tg_i[2]))
            _rp = getattr(smp, 'rpy', None)
            if _rp:
                w("    rpy(标定用): (%.6f, %.6f, %.6f)" % tuple(_rp))
            _n = getattr(smp, 'normal', None)
            if _n is not None:
                w("    法向n(实际): (%.4f, %.4f, %.4f)" % (_n[0], _n[1], _n[2]))
            _bq = getattr(smp, 'board_quat', None)
            w("    板位姿(相机系): pos=[% .5f,% .5f,% .5f]%s"
              % (tt_i[0], tt_i[1], tt_i[2],
                 ("  quat=[%.5f,%.5f,%.5f,%.5f]" % tuple(_bq)) if _bq else ""))
            _ts = getattr(smp, 'ts', None)
            if _ts:
                w("    采集时间: %s" % time.strftime('%Y-%m-%d %H:%M:%S',
                                                time.localtime(_ts)))
        w()
        w("── 写入 perception_calib.py 的形式 ──")
        w("REBOT_HAND_EYE_T_GRIPPER_CAM = np.array([")
        for row in T:
            w("    [% .8f, % .8f, % .8f, % .8f]," % tuple(row))
        w("], dtype=np.float64)")
        w("  ★ 直接存 cv2.calibrateHandEye 返回值 X (gripper<-cam), 不取逆.")
        w("    回代链: T_base_board = T_bg @ X @ T_ct (spread 6.7mm 自洽验证).")
    print("\n✔ 结果已保存: %s" % outpath)
    return outpath


def main():
    rospy.init_node('handeye_sample', anonymous=True, disable_signals=True)
    print(__doc__)
    print("=" * 70)
    print("等待 /aruco_single/pose ...")
    board = BoardPose()

    # 等板位姿
    t0 = time.time()
    while board.fresh() is None and time.time() - t0 < 20:
        time.sleep(0.2)
    if board.fresh() is None:
        print("[错误] 20s 内没有收到 /aruco_single/pose")
        print("  请确认: UI 里勾了「标定模式」并点了「启动相机」,")
        print("  且板子完整出现在画面中(UI 里能看到绿色角点).")
        return 1
    print("✔ 板位姿已就绪")

    # ★★ 2026-10-28: 优先走 /joint_states + FK —— UI 已经占着 CAN 总线,
    #   再起第二个后端必然 "not connected". 只有前者不可用时才回落.
    arm = None
    jsfk = JointStateFK()
    if jsfk.start():
        # 等第一帧 /joint_states
        t0 = time.time()
        while jsfk.fresh() is None and time.time() - t0 < 10:
            time.sleep(0.2)
        r = jsfk.get_pose()
        if r.get('ok'):
            arm = jsfk
            print("✔ 机械臂 FK 已就绪 (来源: /joint_states + pinocchio)")
            print("  当前末端 pos=%s  rpy=%s"
                  % ([round(v, 4) for v in r['pos']],
                     [round(v, 4) for v in r['rpy']]))
            print("  当前关节 =%s" % [round(v, 4) for v in r.get('joints', [])])
        else:
            print("[提示] /joint_states 无数据: %s" % r.get('err'))

    if arm is None:
        print("[提示] 回落到子进程查询 FK(注意: 若 UI 已占用 CAN 会失败)...")
        _sub = ArmFK()
        if _sub.start():
            r = _sub.get_pose()
            if r.get('ok'):
                arm = _sub
                print("✔ 子进程 FK 就绪 pos=%s" % ([round(v, 3) for v in r['pos']]))
            else:
                print("[错误] get_pose 失败: %s" % r.get('err'))
        if arm is None:
            print()
            print("=" * 66)
            print("无法获取机械臂位姿, 手眼标定需要它. 常见原因:")
            print("  1) UI 没启动机械臂 —— 先在 UI 里点「启动机械臂」;")
            print("  2) UI 已占 CAN 总线 —— 这是正常情况, 本脚本会自动走")
            print("     /joint_states+FK 方案; 若该方案也失败, 请看上面的提示.")
            print("=" * 66)
            return 1

    samples = []
    while True:
        try:
            cmd = input("\n[回车=采集 | d=删上一组 | l=列表 | s=保存解算 | q=退出] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break
        elif cmd == 'd':
            if samples:
                samples.pop()
                print("已删除上一组, 剩余 %d 组" % len(samples))
            else:
                print("没有可删除的样本")
        elif cmd == 'l':
            print("已采集 %d 组:" % len(samples))
            for i, (Rg, tg, Rt, tt) in enumerate(samples):
                print("  %02d 末端pos=[% .3f,% .3f,% .3f]  板pos=[% .3f,% .3f,% .3f]"
                      % (i + 1, tg[0], tg[1], tg[2], tt[0], tt[1], tt[2]))
        elif cmd == 's':
            # ★ 数量与姿态覆盖双重门槛
            if len(samples) < 3:
                print("[拒绝] 只有 %d 组, 数学上无法解算(至少 3 组)" % len(samples))
                continue
            if len(samples) < 12:
                print("[警告] 样本偏少(%d 组), 建议 12~20 组才可靠" % len(samples))
            Rs = [smp[0] for smp in samples]
            _max_sep = max((_rot_angle_deg(Rs[i], Rs[j])
                            for i in range(len(Rs)) for j in range(i + 1, len(Rs))),
                           default=0.0)
            if _max_sep < 20:
                print("[警告] 姿态覆盖不足: 样本间最大夹角仅 %.1f°(建议 >30°)." % _max_sep)
                print("        姿态过于单一会导致标定结果严重偏差, 建议继续采不同姿态.")
            if len(samples) < 12 or _max_sep < 20:
                if input("仍要尝试解算? (y/N) ").strip().lower() != 'y':
                    continue
            # ★ 解算前再做一次板位姿稳定性总检
            _stable, _rep = _board_stability_report(samples)
            if _rep:
                print(_rep)
            if not _stable:
                if input("板位姿不稳定, 结果大概率不可用. 仍要解算? (y/N) ").strip().lower() != 'y':
                    continue
            solve_and_save(samples, _default_outpath())
        else:
            # 采集一组
            # ★ 2026-10-30 静止校验: 运动中 /joint_states 是指令插值角,
            #   只有停稳后才是电机实测角. 未停稳则拒绝采样.
            _ok, _dev = arm.check_static()
            if not _ok:
                print("[跳过] 机械臂未停稳(1.5s 内关节最大变化 %s° > 0.15°)."
                      % ("?" if _dev is None else "%.3f" % _dev))
                print("       等臂完全停住(建议停稳后再等 5 秒)重新回车.")
                continue
            bp = board.fresh()
            if bp is None:
                print("[跳过] 板子当前未检出(UI 里应能看到绿色角点)")
                continue
            rp = arm.get_pose()
            if not rp.get('ok'):
                print("[跳过] 机械臂 get_pose 失败: %s" % rp.get('err'))
                continue
            # ★ 板位姿可能已过期: 再取一次确认仍新鲜, 保证与 FK 同时刻
            if board.fresh() is None:
                print("[跳过] 采集瞬间板子丢失, 请保持板子静止可见")
                continue
            bp = board.fresh()

            p = bp.pose.position
            q = bp.pose.orientation
            Rt = quat_to_R(q.x, q.y, q.z, q.w)
            tt = np.array([p.x, p.y, p.z], float)
            Rg = rpy_to_R(*rp['rpy'])
            tg = np.array(rp['pos'], float)

            # ★ 重复位姿检测: 与已有样本姿态太接近的, 提示但不拒绝
            #   (姿态几乎相同的样本对解算【毫无贡献】, 采了等于白采)
            _dup = _nearest_pose_deg(Rg, [smp[0] for smp in samples])
            if _dup is not None and _dup < 8.0:
                print("⚠ 第 %02d 组与已有样本姿态仅差 %.1f° —— 对手眼标定几乎无贡献,"
                      % (len(samples) + 1, _dup))
                print("   建议换一个【明显不同】的姿态(绕不同轴倾斜), 否则可考虑按 d 删除本组")

            # ★ 附加留档字段(与历史 11.txt 对齐, 不参与解算)
            #   法向 n = 末端工具轴在基座系的方向 = R_gripper2base 的第1列
            #   (历史文件里 "法向n(实际)" 就是这个量, 由 rpy 派生)
            _normal = Rg[:, 0].copy()
            samples.append(Sample(
                Rg, tg, Rt, tt,
                joints=[float(v) for v in rp.get('joints', [])],
                normal=_normal,
                rpy=[float(v) for v in rp['rpy']],
                ts=time.time(),
                board_quat=[float(q.x), float(q.y), float(q.z), float(q.w)]))

            dist = np.linalg.norm(tt)
            print("✔ 第 %02d 组: 板距=%.3fm 末端pos=[% .3f,% .3f,% .3f] rpy=[% .1f,% .1f,% .1f]°"
                  % (len(samples), dist, tg[0], tg[1], tg[2],
                     *[math.degrees(v) for v in rp['rpy']]))
            print("   法向n=(% .3f,% .3f,% .3f)  关节=%s"
                  % (_normal[0], _normal[1], _normal[2],
                     "[" + ",".join("%.3f" % v for v in rp.get('joints', [])) + "]"))
            print("   " + _progress_hint(samples))
            # ★ 实时板位姿稳定性检查(每次采样后立即反馈, 不用等采完才发现板在动)
            _stable, _rep = _board_stability_report(samples)
            if _rep:
                print(_rep)

    if samples and len(samples) >= 3:
        if input("\n退出前解算并保存? (Y/n) ").strip().lower() != 'n':
            solve_and_save(samples, _default_outpath())
    arm.stop()
    print("已退出")
    return 0


if __name__ == '__main__':
    sys.exit(main())

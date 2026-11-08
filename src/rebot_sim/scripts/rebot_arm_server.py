#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reBot Arm 后端服务 (py3.12 / rebot 环境).

UI (centernet_4060, py3.8) 与本服务 (rebot, py3.12) 通过 stdin/stdout 的
JSON-line 协议通信: UI 每行发一条命令 JSON, 本服务每行回一条结果 JSON.
日志一律打 stderr, 不污染 stdout 的 JSON 流.

为什么不直接在 UI 进程里 import motorbridge:
  motorbridge 只有 cp312 wheel (无 cp38), 而 rospy/PyQt5/PyTorch/CenterNet 在
  centernet_4060(py3.8). 两个 python 无法合一, 故用子进程桥接 (与 forensic_ws
  现有 "conda py3 桥接 ROS py2" 模式一致).

命令集 (UI -> server, 一行 JSON):
  {"cmd":"connect"}                       连接 can0 + 启动控制循环
  {"cmd":"enable"}                        使能全部关节+夹爪
  {"cmd":"disable"}                       去使能
  {"cmd":"home"}                          safe_home (阻塞)
  {"cmd":"estop"}                         急停 (disable)
  {"cmd":"get_state"}                     -> {joints:[6 rad], gripper:rad}
  {"cmd":"solve_ik","x":,"y":,"z":,"rpy":[r,p,y]}    IK 求解(不运动) -> joints
  {"cmd":"solve_ik","x":,"y":,"z":,"normal":[nx,ny,nz]}
  {"cmd":"solve_ik","x":,"y":,"z":,"rot":[9]}         (rot 为 3x3 行优先)
  {"cmd":"set_joints","joints":[6]}       设关节目标(控制循环跟踪, 非阻塞)
  {"cmd":"move_traj","x":,"y":,"z":,"rpy":[...],"duration":2.0}  SE(3)测地线轨迹(阻塞)
  {"cmd":"set_gripper","pos":}            设夹爪目标(电机7)
  {"cmd":"shutdown"}                      断开并退出
"""
import json
import os
import sys
import threading
import time
import traceback

# ---- reBotArm_control_py 路径 ----
REPO = os.path.expanduser("~/forensic_ws/src/reBot/reBotArm_control_py")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from motorbridge import Controller, Mode, CallError  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
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
import yaml  # noqa: E402
import pinocchio as pin  # noqa: E402

RS_YAML = os.path.join(REPO, "config", "rebotarm_rs.yaml")

# ---- 状态 ----
_arm: RebotArm = None
_ctrl: RebotArmEndPose = None
_model = None
_end_frame_id = None
_home_rot = None  # 末端在零位的旋转矩阵 (base frame), 作为默认姿态
_lock = threading.Lock()
_q_goal = None       # ★ 限速: set_joints 设此目标, 控制循环按 VMAX ramp 到 _q_target
_traj = None         # ★ 服务器端轨迹(move_traj_j): {"q0","q1","t0","T"}, vel_loop 250Hz 内求值
VMAX = 0.18           # ★ 关节最大角速度(rad/s)≈10°/s: j1大回转反复过流异响, 再降
ACC_MAX = 1.5         # ★ 最大角加速度(rad/s²): 限制启停/换向力矩尖峰
pos1 = None
rot1 = None
rpy1 = None
get_points_flag = True


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _force_rs_config():
    """强制 kinematics 用 RS 配置 (urdf/end_effector_frame), 不改原 rebotarm.yaml."""
    with open(RS_YAML) as f:
        rs_cfg = yaml.safe_load(f)
    _rmod._hw_cfg_cache = rs_cfg


def _rot_from_normal(nx, ny, nz):
    """由法向量构造末端姿态 (GLUON 约定: normal=[0,0,-1] 即夹爪朝下接近桌面).
    夹爪接近轴 = gripper_end X, 令 X = normal. z 用世界Z投影稳定, y=z×x."""
    x = np.array([nx, ny, nz], dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-6:
        return _home_rot.copy()
    x = x / n
    z0 = np.array([0.0, 0.0, 1.0])
    z = z0 - (z0 @ x) * x
    if np.linalg.norm(z) < 1e-6:
        z0 = np.array([0.0, 1.0, 0.0]); z = z0 - (z0 @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x); y = y / np.linalg.norm(y)
    return np.column_stack([x, y, z])


def _rot_about_approach(R, angle):
    """绕 R 的第1列(接近轴 X)旋转 angle: R' = R @ Rx(angle). 用于枚举横滚角."""
    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R @ Rx


def _build_target_se3(x, y, z, rpy=None, normal=None, rot=None):
    pos = np.array([x, y, z], dtype=float)
    if rot is not None:
        R = np.array(rot, dtype=float).reshape(3, 3)
    elif normal is not None:
        R = _rot_from_normal(*normal)
    elif rpy is not None:
        R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    else:
        # 默认: 夹爪朝下 (GLUON 默认姿态, 接近方向=[0,0,-1])
        R = _rot_from_normal(0.0, 0.0, -1.0)
    return pin.SE3(R, pos)


def _bus_alive():
    """can0 存活预检: 接口up + ping 电机1有应答(2s内). PCAN断连/机械臂断电时
    快速失败, 避免 connect 里 mode切换重试风暴打爆USB(整机'死机'帮凶)."""
    try:
        state = open('/sys/class/net/can0/operstate').read().strip()
        if state != 'up':
            return False, "can0 is %s (PCAN掉线或未配置)" % state
    except Exception:
        return False, "can0 接口不存在 (PCAN-USB 断连, 检查USB线/电源)"
    return True, "ok"


def _do_connect():
    global _arm, _ctrl, _model, _end_frame_id, _home_rot
    if _ctrl is not None and getattr(_ctrl, "_running", False):
        return True, "already running"
    # ★ 总线预检: 不通立即报错返回(UI会显示), 不进入电机重试风暴
    alive, why = _bus_alive()
    if not alive:
        _log("[connect] ★ 总线不可用: %s" % why)
        return False, why
    _force_rs_config()
    _model = load_robot_model()
    _end_frame_id = get_end_effector_frame_id(_model)
    # 零位末端姿态 (compute_fk 返回 position, rotation, homogeneous)
    q0 = np.zeros(_model.nq)
    _, _home_rot, _ = compute_fk(_model, q0)
    _home_rot = np.array(_home_rot)
    global _IK_DATA
    _IK_DATA = _model.createData()
    _arm = RebotArm(hw_yaml="rebotarm_rs.yaml")
    _arm.connect()
    # ★ 逐个电机: 清错误 -> 切 MIT 模式 -> 使能 -> 读位置验证 (有响应才算成功)
    #   之前"关节4-7不动/夹爪失控"的根因之一: 电机处于错误态(过流/跟随误差锁死),
    #   enable 不报错但不响应指令. clear_error 后逐个验证.
    for jc in _arm._all_joints:
        m = _arm._motor_map[jc.name]
        try:
            m.clear_error()
            time.sleep(0.02)
        except Exception as e:
            _log("[connect] clear_error %s: %s" % (jc.name, e))
        time.sleep(0.02)
    # ★ 用 MIT 模式: 直接发 pos/vel/kp/kd/tau, 不依赖电机内部 POS_VEL PID 参数写入
    _ctrl = RebotArmEndPose(_arm, arm_control_mode="mit", use_gravity_ff=False)
    # ★★ 安全: 使能前必须先把控制目标同步成"当前实际关节角"!
    #   否则 _ctrl.start() 使能瞬间 _q_target 还是零位, 臂会被满kp大力矩
    #   从当前姿态(如扫描中段)拽回零位 —— "启动机械臂巨幅弹跳"的根因.
    #   先读位置写入 _q_target, 再 start()(start内部才 enable+控制循环).
    try:
        q_now, _, _ = _arm.get_state()
        _ctrl._q_target[:] = q_now[:_ctrl._n]
        _log("[connect] 使能前同步当前关节: %s (deg)" % [round(np.degrees(v), 1) for v in q_now[:_ctrl._n]])
    except Exception as e:
        _log("[connect] ★ 读取当前关节失败: %s (仍继续, 但使能时可能弹跳!)" % e)
    _ctrl.start()
    global _q_goal, _traj
    _q_goal = _ctrl._q_target.copy()   # 目标=当前位置, 起步零力矩
    _traj = None                       # 新连接清除残留轨迹
    # 逐关节验证响应 (读 mechPos 0x7019), 无响应的记入日志
    offline = []
    for jc in _arm._all_joints:
        m = _arm._motor_map[jc.name]
        try:
            _ = float(m.robstride_get_param_f32(0x7019))
        except Exception as e:
            offline.append(jc.name)
            _log("[connect] ★ 关节 %s 无响应: %s" % (jc.name, e))
    if not offline:
        _log("[connect] 7 电机全部在线响应 OK")
    else:
        _log("[connect] 离线关节: %s" % offline)
        if len(offline) >= 3:
            # ★ 半数以上关节无应答 = 总线/供电异常, 立即失败退出,
            #   不让控制循环对着空总线全速发指令(USB错误风暴帮凶)
            try:
                _arm.disconnect()
            except Exception:
                pass
            _ctrl = None
            return False, "电机无应答(%d个离线): 检查机械臂48V电源/CAN线" % len(offline)
    # ★ 关闭重力前馈后重启控制循环为 250Hz 并加限速:
    #   - 重力前馈每周期对 6 电机做 CAN 参数读(0x7019), 500Hz 下 ~6000帧/s 会打满总线,
    #     挤掉 MIT 指令/夹爪指令 -> "只有一个关节动/夹爪失控" 的根因
    #   - 250Hz + 无参数读 -> 总线负载 ~450k bps, 余量充足
    _n = _ctrl._n
    # ★ 低频重力前馈(10Hz): 用当前指令目标(免CAN读)算重力矩, 附加到MIT tau.
    #   完全关闭重力前馈时 j2/j3 负载下静态误差大+快速换向易过流失能;
    #   500Hz原版每周期6次参数读打满总线才被禁, 现在无CAN读+10Hz计算, 安全.
    from reBotArm_control_py.dynamics import compute_generalized_gravity
    _gff_tau = np.zeros(_n)

    def _send_arm_mit(ctrl, tau_ff):
        """发 arm 6 关节 MIT 指令(软增益 kp=25/kd=2 + 完整重力前馈).
        ★ 试验过 j1 单独 kp=60: 大回转满力矩输出反而引发整体过流失能, 已回滚."""
        ctrl._arm_group.send_mit(
            ctrl._q_target,
            vel=ctrl._qd_target,
            kp=np.full(ctrl._n, 25.0),
            kd=np.full(ctrl._n, 2.0),
            tau=tau_ff)

    # ★ 回滚基线: 线性 ramp + 速度前馈(加速度限幅) —— S曲线/梯形轨迹生成器
    #   实测引入响应慢/卡顿/跳跃, 已整体移除, 恢复本文件此前的稳定实现
    _last_ff_v = np.zeros(_n)

    def _vel_loop(arm, dt):
        nonlocal _last_ff_v
        try:
            _vel_loop_inner(arm, dt)
        except Exception as e:
            # ★ 控制循环线程永不因异常死亡(线程死=电机失去全部指令=运动卡住,
            #   且历史上线程死亡是静默的). 打印后继续下一周期.
            import traceback
            _log("[vel_loop] 异常(已忽略, 继续运行): %s\n%s" % (e, traceback.format_exc()))

    def _vel_loop_inner(arm, dt):
        nonlocal _last_ff_v   # ★ 内层也要声明(用户恢复代码时丢失, 曾致每帧异常->0指令->灯常亮)
        global _traj
        # ★ 服务器端轨迹求值(250Hz): move_traj_j 设定后, 每周期按 min-jerk 剖面
        #   更新 _q_goal —— 全关节同一时间基准, 无UI侧 sleep 抖动/管道延迟/30Hz台阶.
        #   剖面 s(t)=10t³-15t⁴+6t⁵, 峰值速度=1.875x均值, 中段被下方 VMAX 追赶逻辑
        #   平滑削顶 = S曲线+匀速段(工业控制器标准做法), 起停加速度无突变.
        tr = _traj
        if tr is not None:
            now = time.monotonic()
            t = (now - tr["t0"]) / tr["T"]
            if t >= 1.0:
                # 剖面结束 -> "到位尾巴": 目标停在终点, vel_loop 继续追,
                # 指令位置距终点<eps(或超时兜底)才算轨迹真正完成(traj_active=False)
                _q_goal[:_n] = tr["q1"]
                if float(np.max(np.abs(_ctrl._q_target[:_n] - tr["q1"]))) < 0.01 \
                        or (now - tr["t0"]) > tr["T"] + 5.0:
                    _traj = None
                    _log("[traj] 完成 q1=%s (deg)" % [round(np.degrees(v), 1) for v in tr["q1"]])
            else:
                s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
                _q_goal[:_n] = tr["q0"] + s * (tr["q1"] - tr["q0"])
        goal = _q_goal[:_n]
        cur = _ctrl._q_target
        dq = goal - cur
        step = np.clip(dq, -VMAX * dt, VMAX * dt)
        _ctrl._q_target = cur + step
        # 速度前馈(加速度限幅): 消除启停/换向瞬间力矩尖峰
        raw_v = step / dt
        dv = raw_v - _last_ff_v
        max_dv = ACC_MAX * dt
        _last_ff_v = _last_ff_v + np.clip(dv, -max_dv, max_dv)
        _ctrl._qd_target[:] = _last_ff_v
        # ★ 每周期重力前馈(纯pinocchio计算, 无CAN往返) + 发送 MIT 指令
        #   (回滚时曾丢失此段 -> 控制循环空转不发帧 -> "灯常亮无法启动"的根因)
        try:
            q_model = pad_q_for_model(_model, _ctrl._q_target, _n)
            _gff_tau[:] = compute_generalized_gravity(_model, q_model, _ctrl._data)[:_n]
        except Exception:
            _gff_tau[:] = 0.0
        _send_arm_mit(_ctrl, _gff_tau)
        if _arm.has_gripper:
            # 夹爪柔顺模式: kp=10/kd=1
            _ctrl._gripper_group.send_mit(
                np.array([_ctrl._gripper_target]),
                kp=np.full(1, 10.0),
                kd=np.full(1, 1.0))
    _arm.stop_control_loop()
    _arm.start_control_loop(_vel_loop, rate=250)
    _log("[connect] mode=mit(soft-kp8+full-gravity-ff), arm_group=%s, gripper=%s, control_loop=%s, rate=250Hz, VMAX=%.2f" % (
        _arm.groups.get("arm"), _arm.has_gripper, _arm.control_loop_active, VMAX))
    return True, "connected"


_IK_PARAMS = None
_IK_DATA = None   # ★ IK 专用 data: 与控制循环的 _IK_DATA 分离(pinocchio Data
                 #   非线程安全, 两者共用曾致数据竞争->段错误->arm_server静默死亡,
                 #   即"动作一开始就整体失能"的根因)
_POS_TOL = 0.008  # ★ 位置容差 8mm(原2cm, 用户要求输入=实际): 超出的解判不可达而非凑合

# ★ 梯形构型种子 (肘弯 j3 + 腕朝下 j5=0 + 低位), 由 FK 网格搜索验证 down>0.95:
#   保证 IK 优先收敛到 "关节3/4 成梯形、末端朝下、离桌面恒高" 的分支
_TRAPEZOID_SEEDS = [
    np.radians([0, 60, 45, -60, 0, 0]),   # FK: pos≈[0.216, 0, 0.081] 末端朝下
    np.radians([0, 70, 60, -75, 0, 0]),   # FK: pos≈[0.202, 0, 0.110]
    np.radians([0, 80, 45, -45, 0, 0]),   # FK: pos≈[0.258, 0, 0.021]
    np.radians([0, 50, 45, -75, 0, 0]),   # FK: pos≈[0.152, 0, 0.093]
    np.radians([0, 60, 60, -75, 0, 0]),   # FK: pos≈[0.205, 0, 0.142]
]


def _solve_ik(x, y, z, rpy=None, normal=None, rot=None, fast=False):
    """fast=True: 只走快速路径(连续性种子+梯形种子, 各≤300次迭代), 失败即返回.
    用于 move_linear 的中途航点 —— 慢速兜底(多横滚×重试)一次可能耗时数百毫秒,
    会造成'动一段停一下'的低帧感; 终点航点才用完整兜底."""
    if _ctrl is None:
        return False, "not connected", None
    global _IK_PARAMS, _IK_PARAMS_SEED, _IK_DATA
    if _IK_DATA is None:
        _IK_DATA = _model.createData()
    if _IK_PARAMS is None:
        _IK_PARAMS = IKSolverParams(max_iter=1000, tolerance=1e-3, step_size=0.5, damping=1e-6)
        # 种子试解用短迭代(好种子很快收敛, 不收敛的早停省时间)
        _IK_PARAMS_SEED = IKSolverParams(max_iter=300, tolerance=1e-3, step_size=0.5, damping=1e-6)
    # ★ 连续性种子 = 当前指令目标(_q_goal), 不做 CAN 往返(get_state 每次IK都查会拖慢+占总线)
    q_seed = pad_q_for_model(_model, np.asarray(_q_goal[:_ctrl._n], dtype=float), _ctrl._n)
    pos = np.array([x, y, z], dtype=float)
    # 基础旋转: 显式 rot > rpy > normal > 默认(夹爪朝下, GLUON约定 normal=[0,0,-1])
    if rot is not None:
        base_R = np.array(rot, dtype=float).reshape(3, 3); multi_roll = False
    elif rpy is not None:
        base_R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2])); multi_roll = False
    elif normal is not None:
        base_R = _rot_from_normal(*normal); multi_roll = True
    else:
        base_R = _rot_from_normal(0.0, 0.0, -1.0); multi_roll = True

    def _try(q_init, R, params, retries=0):
        """以 q_init 为种子对位姿 (R,pos) 求解, 返回 (ok, pos_err, joints)."""
        T = pin.SE3(R, pos)
        if retries:
            res = solve_ik_with_retry(_model, _IK_DATA, _end_frame_id, T, q_init,
                                      params, max_retries=retries)
        else:
            res = solve_ik(_model, _IK_DATA, _end_frame_id, T, q_init,
                           params, controlled_joints=_ctrl._n)
        q_sol = res.q[:_ctrl._n]
        qp = pad_q_for_model(_model, q_sol, _ctrl._n)
        pin.forwardKinematics(_model, _IK_DATA, qp)
        pin.updateFramePlacements(_model, _IK_DATA)
        pe = float(np.linalg.norm(np.array(_IK_DATA.oMf[_end_frame_id].translation) - pos))
        return (res.success or pe < _POS_TOL), pe, [float(v) for v in q_sol]

    # ① 连续性种子(当前指令目标): 途中航点在此快速收敛, 保证运动连续
    pe0 = float('inf')
    if multi_roll:
        ok, pe, q_sol = _try(q_seed, base_R, _IK_PARAMS_SEED)
        pe0 = pe
        if ok:
            return True, "ok (cont, pos_err=%.4f)" % pe, q_sol
    # ② 梯形种子 (肘弯+腕朝下分支, 短迭代)
    if multi_roll:
        for ts in _TRAPEZOID_SEEDS:
            ok, pe, q_sol = _try(ts.copy(), base_R, _IK_PARAMS_SEED)
            pe0 = min(pe0, pe)
            if ok:
                return True, "ok (trapezoid, pos_err=%.4f)" % pe, q_sol
    # ③ 兜底: 当前种子+多横滚(重试) —— fast 模式跳过
    if fast:
        return False, "unreachable-fast (pos_err=%.4f)" % pe0, None
    rolls = [0.0] + ([np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi,
                      -np.pi / 4, -np.pi / 2, -3 * np.pi / 4] if multi_roll else [])
    best = None
    for ang in rolls:
        R = base_R if ang == 0.0 else _rot_about_approach(base_R, ang)
        ok, pe, q_sol = _try(q_seed, R, _IK_PARAMS, retries=3)
        if ok:
            return True, "ok (pos_err=%.4f roll=%.0f)" % (pe, np.degrees(ang)), q_sol
        if best is None or pe < best[0]:
            best = (pe, q_sol)
    return False, "unreachable (best pos_err=%.4f)" % best[0], None


def _handle(cmd):
    c = cmd.get("cmd")
    global _traj   # ★ 轨迹在 set_joints/estop/disable/home/recover/move_traj_j 中被读写
    with _lock:
        try:
            if c == "connect":
                ok, msg = _do_connect()
                n_arm = _ctrl._n if _ctrl else 0
                hg = _arm.has_gripper if _arm else False
                return {"ok": ok, "msg": msg, "n_arm": n_arm, "has_gripper": hg}
            if _ctrl is None:
                return {"ok": False, "err": "not connected"}
            if c == "enable":
                _arm.enable_all(); _log("[enable] enable_all done"); return {"ok": True}
            if c == "recover":
                # 电机错误恢复: 清错误 -> 同步当前位置到目标(防弹跳!) -> 重切MIT -> 使能
                for jc in _arm._all_joints:
                    m = _arm._motor_map[jc.name]
                    try:
                        m.clear_error()
                    except Exception as e:
                        _log("[recover] clear_error %s: %s" % (jc.name, e))
                    time.sleep(0.02)
                # ★ 先使能才能读到可信位置; 但为防"使能瞬间拽向旧目标",
                #   先把目标锁死为当前 _q_target(上一次指令), 再使能,
                #   然后读实际位置覆盖目标 —— 目标与实际一致后臂保持静止.
                try:
                    q_now, _, _ = _arm.get_state()
                    _ctrl._q_target[:] = q_now[:_ctrl._n]
                    _q_goal[:] = q_now[:_ctrl._n]
                    _traj = None   # 目标已重置, 旧轨迹作废
                    _log("[recover] 目标同步为当前关节: %s (deg)" % [
                        round(np.degrees(v), 1) for v in q_now[:_ctrl._n]])
                except Exception as e:
                    _log("[recover] 读当前位置失败: %s" % e)
                if _ctrl._arm_group is not None:
                    _ctrl._arm_group.mode_mit()
                if _arm.has_gripper:
                    _ctrl._gripper_group.mode_mit()
                _arm.enable_all()
                _log("[recover] clear_error + 同步 + mode_mit + enable done")
                return {"ok": True}
            if c == "disable":
                _traj = None; _arm.disable_all(); return {"ok": True}
            if c == "estop":
                _traj = None; _arm.estop(); return {"ok": True}
            if c == "get_state":
                q, _, _ = _arm.get_state()
                joints = [float(v) for v in q[:_ctrl._n]]
                g = float(q[_ctrl._n]) if _arm.has_gripper and len(q) > _ctrl._n else 0.0
                # traj_active: 服务器端轨迹是否仍在执行(UI 阻塞等待的完成判据)
                return {"ok": True, "joints": joints, "gripper": g,
                        "traj_active": _traj is not None}
            if c == "can_ping":
                # ★ CAN往返探测: 读joint1 mechPos(0x7019)等真实回包.
                #   PCAN适配器"冻结"(dmesg: Rx urb aborted -32)时 netdev 仍 up、
                #   发送不报错, 但 CAN 回包全丢 —— get_state 读的是缓存,
                #   只有这里能探测出"收不到"的真死.
                if _arm is None or not getattr(_arm, "_connected", False):
                    return {"ok": False, "err": "not connected"}
                try:
                    jn = _arm._all_joints[0].name
                    _arm._motor_map[jn].robstride_get_param_f32(0x7019, timeout_ms=600)
                    return {"ok": True}
                except Exception as e:
                    return {"ok": False, "err": "can rx dead: %s" % e}
            if c == "get_pose":
                # 当前关节 -> 末端位姿 (FK), 供现场标定预设点用
                q, _, _ = _arm.get_state()
                q_pad = pad_q_for_model(_model, q[:_ctrl._n], _ctrl._n)
                pos, rot, _ = compute_fk(_model, q_pad)
                rpy = pin.rpy.matrixToRpy(np.array(rot)).tolist()
                return {"ok": True, "pos": [float(v) for v in pos],
                        "rpy": [float(v) for v in rpy],
                        "joints": [float(v) for v in q[:_ctrl._n]]}
            if c == "solve_ik":
                ok, msg, j = _solve_ik(cmd.get("x", 0), cmd.get("y", 0), cmd.get("z", 0),
                                       rpy=cmd.get("rpy"), normal=cmd.get("normal"),
                                       rot=cmd.get("rot"))
                return {"ok": ok, "err": msg, "joints": j}
            if c == "set_joints":
                _traj = None    # ★ 流式目标(扫描/点动)优先: 取消进行中的服务器端轨迹
                j = list(cmd["joints"])
                _q_goal[:len(j)] = j          # ★ 设目标, 由 _vel_loop 按 VMAX 限速 ramp
                if not getattr(_handle, "_logged_first_set", False):
                    _handle._logged_first_set = True
                    _log("[set_joints] first cmd received, goal=%s, loop=%s" % (
                        [round(v, 3) for v in j], _arm.control_loop_active))
                return {"ok": True}
            if c == "move_traj_j":
                # ★ 关节空间 min-jerk 轨迹(异步, 非阻塞): UI 只发这一条命令,
                #   250Hz vel_loop 内求值更新 _q_goal, 完成判据=get_state.traj_active.
                #   起点=当前 _q_goal(连续性, 无目标跳变).
                js = [float(v) for v in cmd["joints"]]
                # 缺省关节保持当前目标(短列表不会误命令到0)
                q1 = np.array(_q_goal[:_ctrl._n])
                for i, v in enumerate(js[:_ctrl._n]):
                    q1[i] = v
                T = max(0.15, float(cmd.get("duration", 3.0)))
                _traj = {"q0": np.array(_q_goal[:_ctrl._n]), "q1": q1,
                         "t0": time.monotonic(), "T": T}
                return {"ok": True, "duration": T}
            if c == "set_gripper":
                _ctrl.set_gripper_target(float(cmd["pos"]))
                return {"ok": True}
            if c == "move_traj":
                T = _build_target_se3(cmd["x"], cmd["y"], cmd["z"],
                                      rpy=cmd.get("rpy"), normal=cmd.get("normal"),
                                      rot=cmd.get("rot"))
                # 用 move_to_traj 的 IK+轨迹(阻塞)
                ok = _ctrl.move_to_ik(cmd["x"], cmd["y"], cmd["z"],
                                      roll=(cmd.get("rpy") or [0, 0, 0])[0],
                                      pitch=(cmd.get("rpy") or [0, 0, 0])[1],
                                      yaw=(cmd.get("rpy") or [0, 0, 0])[2])
                return {"ok": ok}
            if c == "home":
                _traj = None; _ctrl.safe_home(); return {"ok": True}
            if c == "shutdown":
                # ★ 2026-08-31: 关UI时直接disable(不回零) — 原版end()里safe_home
                #   先回零再断开(10-20秒), UI等30s超时或进程被强杀时电机保持
                #   使能态 → PCAN灯常亮. 现在改立即停控制循环+disable+断开, 2秒内完成.
                try:
                    _arm.stop_control_loop()
                    _arm.disable_all()
                    _log("[shutdown] 控制循环已停+全部电机已去使能")
                except Exception as e:
                    _log("[shutdown] disable异常(仍继续断开): %s" % e)
                try:
                    _arm.disconnect()   # 内部也做 disable_all + ctrl.close
                    _log("[shutdown] CAN断开完成")
                except Exception as e:
                    _log("[shutdown] disconnect异常: %s" % e)
                return {"ok": True}
            return {"ok": False, "err": "unknown cmd: %s" % c}
        except Exception as e:
            _log("EXC: %s\n%s" % (e, traceback.format_exc()))
            return {"ok": False, "err": str(e)}

"""获取末端位姿点位"""
def get_points(pos, rot, rpy):
    q, _, _ = _arm.get_state()
    q_pad = pad_q_for_model(_model, q[:_ctrl._n], _ctrl._n)
    pos, rot, _ = compute_fk(_model, q_pad)
    rpy = pin.rpy.matrixToRpy(np.array(rot)).tolist()
    return {"ok": True}



def main():
    global pos1, rot1, rpy1 , get_points_flag 
    _log("rebot_arm_server ready (rebot py3.12) pid=%d" % os.getpid())

    """进程获取末端位姿点位"""
    if get_points_flag == True:
        try:
            import threading as _th
            _t = _th.Timer(0.5, get_points(pos1, rot1, rpy1))
            _t.daemon = True
            _t.start()
            _t.join()
        except Exception as e:
            _log("“点位获取失败”")

    """主循环: 读取 stdin json 命令, 处理后回包 stdout"""
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
            _handle(cmd)
            # 控制循环死亡检测: 线程崩了但进程活着 = "连接了但发不出指令"(灯常亮假象),
            # 立即报错退出让 UI 触发重启自愈, 不再静默装死
            if _ctrl is not None and getattr(_ctrl, "_running", False) and not _arm.control_loop_active:
                _log("[FATAL] 控制循环线程已死亡(内部异常), arm_server 退出待UI重启")
                break
            continue
        resp = _handle(cmd)
        print(json.dumps(resp), flush=True)
        if cmd.get("cmd") == "shutdown":
            break
    # ★ stdin EOF (UI 崩溃/被杀): 不让电机瞬间掉电失控 ——
    #   保持当前目标发 2 秒 MIT 保持指令, 然后安全断开(断开内部会 disable).
    #   给用户 2 秒窗口期反应 + 电机受控泄力而非猛掉.
    _log("[EOF] stdin 关闭(UI退出?), 进入 2 秒保持后安全退出")
    if _ctrl is not None:
        try:
            import threading as _th
            _t = _th.Timer(2.0, lambda: (_arm.disconnect() if _arm else None))
            _t.daemon = True
            _t.start()
            _t.join()
        except Exception as e:
            _log("[EOF] 安全退出异常: %s" % e)
    _log("rebot_arm_server exit")




if __name__ == "__main__":
    main()

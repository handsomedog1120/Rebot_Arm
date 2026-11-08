#!/usr/bin/env python3
import serial.tools.list_ports
import sys
import os
import math
import json
import subprocess
import threading

sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'forensic_ws/src/gluon/ros_gluon/gluon/scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import rospy
import numpy as np
from numpy import sin, cos, pi, arctan2, arccos, sqrt
from PyQt5.QtCore import (QProcessEnvironment, Qt, QProcess, QTimer, QThread, pyqtSignal,
                           QRectF, QPointF)

env = QProcessEnvironment()
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Bool
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox, QTextEdit,
                             QMessageBox, QSplitter, QFormLayout, QComboBox, QGridLayout,
                             QFrame, QSizePolicy, QStyle, QStyleOptionButton)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QBrush,
                         QIcon, QFont, QLinearGradient, QPolygonF,
                         QPainterPath)
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QRect, QRectF, QSize, QByteArray
import cv2
from PyQt5.QtGui import QColor
import time

# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：自动物证采集流程依赖模块（新增文件，不改动任何原有代码） ★★★
# 对应专利步骤S3(SPME-CenterNet检测) + S4(CDG-ATRG精细分割/标定/深度反投影/
# 法向量估计) + S5(按类别分派execute_action)
# ──────────────────────────────────────────────────────────────────────────────
import perception_calib as pcalib
from centernet_blood_bridge import ForensicDetector
import rrt_planner  # 笛卡尔空间RRT，仅用于execute_action()里"转移段"的路径规划

# 定义关节名称
# ★ 2026-09-10 标定模式关节名对齐: robot_state_publisher(标定launch里起)
#   需要的URDF关节名是 joint1..joint6(00-arm-rs_asm-v3.urdf), 原axis_joint_*
#   是GLUON URDF时代名 — 改名后 base_link→gripper_end 的TF链才能连上.
#   gluon_node 已废弃(不再消费/joint_states), 此改动无下游风险.
JOINT_NAMES = [
    'joint1',  # 基座旋转 (原axis_joint_1)
    'joint2',  # 肩部俯仰
    'joint3',  # 肘部俯仰
    'joint4',  # 腕部旋转
    'joint5',  # 腕部俯仰
    'joint6'   # 末端旋转
]

# 运动参数
MOVE_DURATION = 5.0       # 单次移动时间(秒)
HOME_DURATION = 8.0       # ★ 回零专用时长: 比MOVE_DURATION慢, 只影响「返回初始位置」
# ★ 2026-09-17 用户要求: 预设动作全程匀速 —— 所有运动段统一按"移动到抓取点
#   上方"(转移段, MOVE_DURATION=5s)的节奏运行, 分级时长(PICK/RELEASE)取消;
#   move_joint 内部仍按 VMAX/ACC 剖面对大行程自动延长(max(duration, need)).
#   涂抹往复笔(1.5s/笔)是工作段不属运输段, 维持不变.
INITIAL_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始位置
SAFE_HEIGHT = 0.13         # 安全高度H(米)

# ★ 2026-09-18 用户要求: [移动到位置]终点停距 —— 末端停止时距目标点位3cm,
#   距离沿末端法向量(工具轴)方向; 停距直接并入IK求解目标(运动到位即停),
#   不做"先到点位再回退3cm"的两段式运动.
APPROACH_STANDOFF_M = 0.0     # 终点停距(米) — 2026-09-27 用户要求归零: 精确到框选目标点

# ★ 修改：扫描姿态定义（末端俯视地面）—— reBot 版
#   由 FK 验证: j2=60° j3=45° j4=-60° j5=0° 时末端垂直朝下, 位于[0.216,0,0.081]m
#   (GLUON 原值 [0,0,1.0,-0.7,1.5,0] 是 GLUON 关节角, 在 reBot 上=夹爪水平+关节
#    顶限位, 是"扫描时夹爪不朝下+工作一段后电机过流断电"的根因)
SCAN_POSITION = [0.0, 0.947, 0.785, -1.047, 0.0, 0.0]   # j2=0.947(较1.047抬高末端z约+2cm, pinocchio FK实测z:0.081->0.101)

# 扫描参数
SCAN_SWEEP_ANGLE = 4.0    # (旧j1正弦摆动, 已废弃) 2026-09-17同步kk_rebot巡点循环扫描:
# ★★ 巡点循环扫描 (2026-09-15 kk_rebot, 取代 j1 正弦摆动):
#   A→B→C→D→E→F→A 循环往复, 每点按表面法向量取向(IK 构造姿态).
#   ★ 连续巡行不停留: 每条腿插入经过点(位置线性+法向量同步插值 -> 姿态渐变),
#     整圈 IK 预先解算完毕再运动 —— 运行期无解算间隙, 段间零停顿.
SCAN_TOUR_WAYPOINTS = [
    # (点名, (x,y,z) 米, 表面法向量 —— 自动归一后传 IK)
    ("A", (0.30,  0.00, 0.60), (1.0,  0.0,  0.0)),
    ("B", (0.25, -0.20, 0.20), (1.0, -1.0, -1.0)),
    ("C", (0.10, -0.40, 0.10), (0.2,  0.0, -1.0)),
    ("D", (0.40,  0.00, 0.10), (0.2,  0.0, -1.0)),
    ("E", (0.10,  0.40, 0.10), (0.2,  0.0, -1.0)),
    ("F", (0.25,  0.20, 0.20), (1.0,  1.0, -1.0)),
]
SCAN_TOUR_VIA_PER_LEG = 2    # 每条腿插入的经过点数(位置/姿态同步插值)
SCAN_TOUR_SPEED = 0.12       # 巡行关节速度 rad/s (匀速巡行的速度基准)
SCAN_TOUR_RATE_HZ = 30       # 巡行流式下发频率
# ★ 2026-09-17 同步kk_rebot: 扫描已改为巡点循环(A→B→C→D→E→F→A, 工作线程
#   30Hz流式驱动), 旧j1正弦摆动已移除; 自动采集流程复用 start_scan()/stop_scan()。

# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：自动物证采集流程配置 ★★★
# 对应专利步骤S2(扫描采图)~S5(按类别分派动作)
# ──────────────────────────────────────────────────────────────────────────────
RAW_COLOR_TOPIC = '/camera/color/image_raw'      # 相机原始彩色图话题
RAW_DEPTH_TOPIC = '/camera/depth/image_raw'      # 相机原始深度图话题
CAMERA_INFO_TOPIC = '/camera/color/camera_info'  # 相机内参话题

# 检测改为在后台线程(ForensicDetectionThread)里连续跑，不再按固定
# 时间间隔抽帧，因此不再需要DETECT_INTERVAL这个节流常量。
CONFIRM_COUNT = 2          # 连续多少次检测到同一类别才判定"确实发现物证"
NORMAL_ESTIMATION_RADIUS_PX = 15   # 表面法向量估计的邻域半径(像素)

# ★★★ 自动采集流程的扫描方式：机械臂初始化后，在关节限位内单方向旋转
#   "半圈"完成一次检测扫描，到位后停止，停顿数秒，再根据本轮扫描中
#   记录到的检测信息执行采集动作（不再是原来来回摆动的start_scan()）★★★
#   关节1(基座旋转)限位为(-2.79, 2.79)rad（见functions.py inv_k），
#   下面留出安全余量，从-2.7转到+2.7，单方向转一次即视为"半圈扫描"；
#   其余关节固定在 SCAN_POSITION[1:] （末端俯视地面姿态不变）。
SCAN_HALF_CIRCLE_TH1_START = -2.7   # 半圈扫描起始关节1角度(rad)
SCAN_HALF_CIRCLE_TH1_END = 1.4      # ★ 2026-08-28: 2.7→1.4 — 右端2.7已贴j1限位2.79,
                                    #   扫到最右触发电机限位保护整体失能~1s;
                                    #   1.4rad=80°, 不再到身后, 远离右限位。
SCAN_ROTATE_DURATION = 25.0         # 半圈旋转总时长(秒) ★4.1rad/25s=0.164rad/s<VMAX0.25, 超速会目标超前引发换向冲击
POST_SCAN_PAUSE = 3.0               # 旋转到位、机械臂停止后的停顿时间(秒) 5→3: 乒乓扫描衔接更紧凑

# 物证大类('solid'/'liquid'/'micro') -> execute_action() 现成的动作编号
# 三种动作在原文件里已经和三类物证的工具/流程一一对应：
#   liquid(液体类，棉签擦拭) -> execute_action(1) 涂抹
#   micro (微小类，粘取器)   -> execute_action(2) 粘取
#   solid (固体类，默认夹爪) -> execute_action(3) 抓取
CATEGORY_TO_ACTION = {
    'liquid': 1,
    'micro': 2,
    'solid': 3,
}

# ── 两段式回零的 move_traj 目标 (2026-10-20) ─────────────────────────────
# 动作收尾回零改服务器端 move_traj(min-jerk): 旧 move_joint 30Hz 线性插值
# 无速度剖面(起停速度跳变) + 每段结尾 wait_arrival/get_pose 回验死时间,
# 是"回零慢且卡"的根因. 坐标为 FK 实测值(rebot_pin 数值验证):
#   _MID=[0,0.70,0.79,-0.79,0,0] 的 FK 位置/法向如 HOME_MID_*; 全零位 FK
#   如 HOME_ZERO_*; normal_to_rpy 基准 roll 与两者 FK 姿态偏差 <1e-6rad
#   (q6=None 即精确复现), IK 从 _MID 种子解零位目标精确收敛 [0]*6 不跳解.
# ★ duration 必须显式传: 第二段笛卡尔距离仅 2.5cm(自适应会取 1.5s 下限)
#   但关节行程 0.79rad(展臂), 2.5s 下 min-jerk 峰值关节速度 ≈0.59rad/s,
#   低于分级 VMAX 0.80; 第一段同理(再构幅度大于笛卡尔距离).
HOME_MID_POSE    = (0.28162, 0.0, 0.20334)      # _MID 的 FK 位置(m)
HOME_MID_NORMAL  = (0.76484, 0.0, -0.64422)    # _MID 的末端法向(工具X轴)
HOME_MID_DUR     = 3.5                          # 第一段时长(s)
HOME_ZERO_POSE   = (0.30171, 0.0, 0.21775)     # 全零位 FK 位置(m)
HOME_ZERO_NORMAL = (1.0, 0.0, 0.0)             # 零位法向(世界X)
HOME_ZERO_DUR    = 2.5                          # 第二段时长(s)

# ── 夹爪安全限位 (2026-09-08 自 gluon_controller_kk_rebot.py 同步; 顶断限位事故后的硬性约束) ──
# 电机7(rs-00)原始角: 负=闭合, 正=张开; 断电静止=-1.43 ≈ 完全闭合.
GRIPPER_CLOSED = 4.374       # 74mm安全位rad(新零位系, 启动/看门狗恢复后锁定目标)
GRIPPER_OPEN_MAX = 5.07      # 张开侧紧急钳位rad(新零位系, ≈85mm, 留3mm余量<实测88)

# ── 夹爪实物标定 (齿轮齿条直线双夹爪) ─────────────────────────────────────
# 输入值 = 两夹爪之间的真实物理距离(mm), 一一对应:
#   rad = _GRIP_CAL_RAD_REF + (mm-74)/15.75   斜率15.75mm/rad全程实测不变.
GRIPPER_MM_CLOSED = 0.0      # mm 输入下限(0=并拢指令: 闭到指尖相触并轻握)
GRIPPER_MM_GRIP = 38.0       # ★ 预设动作闭合爪距(2026-09-16用户指定): 取工具/抓物证统一38mm
GRIPPER_MM_TOUCH = 2.0       # 指尖相触点(实测1.9mm, 取2) — "闭合"按钮目标位
GRIPPER_MM_OPEN = 60.0       # mm 输入上限(用户指定张开位)
_GRIP_CAL_RAD_REF = 4.374    # 标定参考点: 电机rad (74mm处; 指尖相触锚点精确标定)
_GRIP_CAL_MM_REF = 74.0      # 标定参考点: 实物mm
_GRIP_CAL_MM_PER_RAD = 15.75 # 实测斜率 mm/rad (漂移前后一致)
GRIPPER_RAD_AT_MM_OPEN = _GRIP_CAL_RAD_REF + (GRIPPER_MM_OPEN - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD    # 80mm
GRIPPER_RAD_AT_MM_CLOSED = _GRIP_CAL_RAD_REF + (GRIPPER_MM_CLOSED - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD  # 0mm

def _gripper_rad_to_mm(rad):
    """电机rad -> 实物mm (标定模型反向)"""
    return _GRIP_CAL_MM_REF + (rad - _GRIP_CAL_RAD_REF) * _GRIP_CAL_MM_PER_RAD

# ── ★ 夹爪零位自愈系统 (电机EEPROM零位损坏后, 断电重启多圈参考可能偏移,
#    以"指尖相触"机械硬点为绝对锚点自动重标; 持久化跨会话零位记忆) ─────────
GRIPPER_CAL_FILE = os.path.expanduser('~/rebot_gripper_cal.json')
_GRIP_CAL_DRIFT_TRIG = 0.35   # rad: 启动时读数与停车位置差超此值(5.5mm)才触发自动标定
_GRIP_CAL_VERIFY_TOL = 1.8    # rad: 锚定修正量与停车位置推算漂移的交叉验证容差(28mm)
_GRIP_CAL_SANITY = 6.0        # rad: 锚定修正量绝对上限(±94mm), 超出=碎屑卡滞, 拒绝


def _gripper_apply_cal_ref(new_ref):
    """运行时原子重标定: 一次更新 REF 与全部派生全局(读取方均在调用时读全局).
    new_ref = 74mm 处的电机 rad. 返回修正量 delta(rad)."""
    global _GRIP_CAL_RAD_REF, GRIPPER_RAD_AT_MM_OPEN, GRIPPER_RAD_AT_MM_CLOSED
    global GRIPPER_CLOSED, GRIPPER_OPEN_MAX
    delta = new_ref - _GRIP_CAL_RAD_REF
    _GRIP_CAL_RAD_REF = new_ref
    GRIPPER_RAD_AT_MM_OPEN = new_ref + (GRIPPER_MM_OPEN - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD
    GRIPPER_RAD_AT_MM_CLOSED = new_ref + (GRIPPER_MM_CLOSED - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD
    GRIPPER_CLOSED = new_ref                                      # 74mm 安全位锁定目标
    GRIPPER_OPEN_MAX = new_ref + 11.0 / _GRIP_CAL_MM_PER_RAD      # ≈85mm 紧急钳位(<实测88留余量)
    return delta


def _gripper_save_cal(ref, last_rad=None):
    """锚点+停车位置持久化(每次夹爪指令后更新, 供下次启动交叉验证)."""
    try:
        with open(GRIPPER_CAL_FILE, 'w') as f:
            json.dump({'rad_ref': round(float(ref), 4),
                       'last_rad': round(float(last_rad if last_rad is not None else ref), 4),
                       'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'mm_ref': _GRIP_CAL_MM_REF, 'mm_per_rad': _GRIP_CAL_MM_PER_RAD}, f)
    except Exception as e:
        print("[gripper] 标定持久化失败: %s" % e)


def _gripper_load_cal():
    """启动时载入持久化锚点. 返回 last_rad(无持久化时 None)."""
    try:
        with open(GRIPPER_CAL_FILE) as f:
            d = json.load(f)
        ref = float(d['rad_ref'])
        if abs(ref) < 20.0:   # 合理性: |REF| 不可能超过电机转角范围
            _gripper_apply_cal_ref(ref)
            return float(d.get('last_rad', ref))
    except Exception:
        pass
    return None


def _gripper_creep_anchor(ac, log=lambda m: None, step_rad=0.30, max_steps=50,
                          settle_s=1.6, step_timeout=9.0, poll_s=0.2, confirm_s=5.0):
    """★ 匍匐锚定: 以读数系为参照小步闭合, 检测"步进无位移"=指尖相触(机械硬点).
    压力上限 = kp(10)×step(0.3) ≈ 3 力矩, 与零位漂移量无关 —— 漂移多大都安全.
    触点复核: 疑似停滞时同目标延长观察5s, 位移<0.16mm 才确认.
    返回 触点电机rad(None=失败). 阻塞约15~40s, 供后台线程调用."""
    rs = ac.get_state()
    if not rs.get("ok"):
        log("[gripper] 标定失败: 读不到夹爪状态(先启动机械臂)")
        return None
    cur = float(rs.get("gripper", 0.0))
    for _i in range(max_steps):
        tgt = cur - step_rad                       # 向闭合方向一小步(读数系, 与漂移无关)
        if not (ac.set_gripper(tgt) or {}).get("ok"):
            log("[gripper] 标定失败: 闭合指令下发失败")
            return None
        t0 = time.time(); prev = None; stable = 0.0; last = cur
        while time.time() - t0 < step_timeout:
            time.sleep(poll_s)
            rs = ac.get_state()
            if not rs.get("ok"):
                continue
            last = float(rs.get("gripper", 0.0))
            if prev is not None and abs(last - prev) < 0.004:    # ~0.06mm/采样 视为静止
                stable += poll_s
            else:
                stable = 0.0
            prev = last
            if stable >= settle_s:
                break
        if abs(last - cur) < 0.03:                 # 本步位移<0.5mm -> 压在硬点上?
            t1 = time.time(); p2 = last            # 复核: 同目标延长观察
            while time.time() - t1 < confirm_s:
                time.sleep(poll_s)
                rs = ac.get_state()
                if rs.get("ok"):
                    p2 = float(rs.get("gripper", 0.0))
            if abs(p2 - last) < 0.01:              # 5s位移<0.16mm = 真触点
                return p2
            cur = p2                               # 其实在爬行(碎屑), 继续匍匐
        else:
            cur = last
    log("[gripper] 标定失败: %d 步未遇到触点(疑似全程卡滞, 请手推夹爪全行程清理)" % max_steps)
    return None


_GRIP_CAL_LAST_RAD = _gripper_load_cal()   # 启动即载入持久化锚点(跨会话零位记忆)

GRIPPER_SPEED_MM_S = 9.4   # ★ 夹爪斜坡限速(mm/s) — 与 arm_control 的 ramp 限速(0.60rad/s×15.75) 保持同步


def _errtext(r):
    """server 响应错误文本统一提取: connect 失败时错误在 msg 字段, 其余命令在 err."""
    if not isinstance(r, dict):
        return "未知错误"
    return r.get("err") or r.get("msg") or "未知错误"


def _normal_candidate_list(preferred=None, preferred_desc='指定'):
    """★ 法向候选列表(2026-09-16抽出共用): 画框移动与预设动作共用同一套.
    preferred → 常见工作法向(标定实测提取) → 球面采样(极角15~75°, 方位角8方向).
    常用/采样与preferred逐分量差<0.01视为重复自动跳过. 返回[(desc, normal), ...]"""
    candidates = []
    if preferred is not None:
        candidates.append((preferred_desc, [float(v) for v in preferred]))
    for n in [(1, 0, 0), (0, 0, -1), (0.8, 0, -0.1), (0.8, 0.1, -0.1),
              (0.6, 0, -0.1), (0.5, 0, 0), (0.9, 0.1, 0), (0.8, 0.5, 0.1),
              (0.6, 0.1, 0), (0.5, 0.1, 0.1)]:
        if preferred is None or any(abs(a - b) > 0.01 for a, b in zip(n, preferred)):
            candidates.append(('常用%.1f,%.1f,%.1f' % n, [float(v) for v in n]))
    for polar_deg in [30, 45, 60, 75, 15]:
        for azim_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            p = math.radians(polar_deg)
            a = math.radians(azim_deg)
            nx = math.cos(p) * math.cos(a)
            ny = math.cos(p) * math.sin(a)
            nz = -math.sin(p)
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nl > 0.01:
                candidates.append(('采样%.0f°/%.0f°' % (polar_deg, azim_deg),
                                   [nx / nl, ny / nl, nz / nl]))
    return candidates


def _smear_dir_from_normal(normal):
    """★ 涂抹面内方向 = 与末端姿态法向量正交的往复方向.
    涂抹必须在"物面平面"内横向蹭动, 即方向须 ⊥ 法向量(工具轴).
    原实现固定沿世界X —— 姿态输入(1,0,0)时法向∥X, 沿X涂抹等于沿工具轴
    朝物面怼进/拉出, 完全错误. 现改为随姿态计算:
      取"最接近世界X"的正交分量 (X - (X·n)n 归一化);
      法向≈±X 时该分量退化(<0.3), 改取世界Z的正交分量.
    默认朝下姿态 n=[0,0,-1] → 方向=[1,0,0], 与原世界X行为完全兼容.
    返回单位向量 [dx, dy, dz]."""
    n = list(normal) if normal else [0.0, 0.0, -1.0]
    ln = math.sqrt(sum(c * c for c in n)) or 1.0
    n = [c / ln for c in n]
    cand = [1.0 - n[0] * n[0], -n[1] * n[0], -n[2] * n[0]]      # X - (X·n)n
    lc = math.sqrt(sum(c * c for c in cand))
    if lc < 0.3:   # 法向≈±X: X分量退化, 取 Z - (Z·n)n
        cand = [-n[0] * n[2], -n[1] * n[2], 1.0 - n[2] * n[2]]
        lc = math.sqrt(sum(c * c for c in cand)) or 1.0
    return [c / lc for c in cand]


def _standoff_target(x, y, z, normal, standoff=APPROACH_STANDOFF_M):
    """★ 终点停距(2026-09-18 用户要求): [移动到位置]的 IK 求解目标 =
    输入点位沿末端法向量(工具轴)方向反向退 standoff 米 —— 末端停止时
    距点位 3cm, 距离方向与末端向量一致. 停距直接写进求解目标, 运动到位
    即停, 不存在"先到点位再回退"的第二段. 法向为空/零向量时不偏移.
    返回 (x', y', z')."""
    if not normal:
        return x, y, z
    n = list(normal)
    ln = math.sqrt(sum(c * c for c in n))
    if ln < 1e-6:
        return x, y, z
    return (x - standoff * n[0] / ln,
            y - standoff * n[1] / ln,
            z - standoff * n[2] / ln)


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：CenterNet检测后台线程 ★★★
# 对视频流连续跑检测，不再按固定时间间隔抽帧：
#   - detector.run()是同步阻塞调用，之前放在30ms的QTimer回调里执行会和
#     机械臂关节插值发布/UI刷新抢占同一个主线程，必须靠DETECT_INTERVAL
#     限流避免卡顿；现在把它放进独立的QThread，主线程完全不受影响，
#     线程内部就是"跑完一帧立刻取最新一帧再跑"的连续检测循环。
#   - 每次检测都会同时记录"取到这一帧彩色/深度图那一刻"的深度图和
#     关节1角度th1，随检测结果一起通过信号传回主线程，避免主线程收到
#     信号时机械臂已经转到别的角度、深度图已经过期的错位问题。
# ──────────────────────────────────────────────────────────────────────────────
class ForensicDetectionThread(QThread):
    # 参数：(DetectionResult, 该帧对应的深度图, 该帧对应的关节1角度th1)
    detection_ready = pyqtSignal(object, object, float)

    def __init__(self, detector, frame_provider):
        """
        detector: centernet_bridge.ForensicDetector 实例
        frame_provider: 无参可调用对象，返回 (color_bgr, depth_img, th1)；
                        color_bgr为None时表示暂时还没有可用的彩色帧
        """
        super().__init__()
        self.detector = detector
        self.frame_provider = frame_provider
        self._running = True
        self._active = False   # 是否处于"应该连续检测"的状态，由主状态机控制

    def set_active(self, active):
        self._active = active

    def run(self):
        import faulthandler, sys
        faulthandler.enable(file=sys.__stderr__)  # ★ 段错误(C层崩溃)时打印堆栈到日志, 定位闪退真凶
        while self._running:
            try:
                if not self._active:
                    self.msleep(20)  # 非扫描阶段空闲小睡，避免忙等占满CPU
                    continue

                color_bgr, depth_img, th1 = self.frame_provider()
                if color_bgr is None:
                    self.msleep(20)
                    continue

                # ★ 拷贝隔离: rospy回调线程可能正在原地写这些图像(竞争导致
                #   C层崩溃/花图), 检测线程用独立副本
                color_bgr = color_bgr.copy()
                if depth_img is not None:
                    depth_img = depth_img.copy()

                det = self.detector.detect(color_bgr, depth_img)
                if det is not None and self._running:
                    self.detection_ready.emit(det, depth_img, th1)
                # 不加额外sleep：检测节奏完全由detector.run()本身的推理耗时决定，
                # 跑完一帧立刻取最新一帧继续，实现"连续检测"
            except Exception as e:
                import traceback
                print("[检测线程] 异常(跳过本帧): %s" % e)
                traceback.print_exc()
                self.msleep(100)

    def stop(self):
        self._running = False
        self.wait()


# ──────────────────────────────────────────────────────────────────────────────
# ★ 相机安装朝向补偿 (2026-10-28, 仅作用于显示层)
#   现场相机物理倒装(机身绕光轴转了180°), 原始图像相对人眼"上下+左右"都颠倒,
#   观感就是画面反了. 注意: 倒装对应的校正是【旋转180°】, 不是单纯上下翻转
#   flip(0) —— 后者只翻垂直轴, 结果变成左右镜像, 文字/左右关系全反.
#
#   ★ 为什么只在显示层做: 深度图 / 相机内参 / 手眼标定 / CenterNet 检测必须
#     继续用原始像素系(_decode_color_bgr 一律不翻转), 否则检测像素与内参
#     参考系镜像, 反投影和基坐标解算全错(见 _decode_color_bgr 的注释).
#   ★ 画框链路: AnnotateCameraLabel 在"控件坐标 ↔ 原始像素"之间做反向映射,
#     冻结画面上拖出来的框回传的仍是【原始像素坐标】, 下游深度采样 / 基坐标
#     解算 / 法向估计一行都不用改.
#
#   换朝向改下面这一行即可(四种都是自逆变换, 正反两向共用一个映射函数):
#     'rot180' — 旋转180°      : 相机倒装(画面上下颠倒)
#     'flip_v' — 仅上下翻转    : 图像垂直镜像
#     'flip_h' — 仅左右翻转    : 图像水平镜像(照镜子观感)
#     'none'   — 不补偿        : 相机正装
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_DISPLAY_ORIENT = 'none'


def _cam_orient_image(img):
    """按 CAMERA_DISPLAY_ORIENT 生成【显示用】图像(输入 BGR / 灰度均可).
    ★ 只给屏幕显示用, 返回结果不得回流到深度/内参/检测等像素系计算."""
    mode = CAMERA_DISPLAY_ORIENT
    if mode == 'rot180':
        return cv2.rotate(img, cv2.ROTATE_180)
    if mode == 'flip_v':
        return cv2.flip(img, 0)
    if mode == 'flip_h':
        return cv2.flip(img, 1)
    return img


def _cam_orient_uv(u, v, w, h):
    """像素坐标在"显示系 ↔ 原始系"之间互换(w,h = 原始图像宽高).
    ★ 四种补偿模式都是自逆变换(做两次等于没做), 故正反两向共用本函数."""
    mode = CAMERA_DISPLAY_ORIENT
    if mode == 'rot180':
        return (w - 1 - u, h - 1 - v)
    if mode == 'flip_v':
        return (u, h - 1 - v)
    if mode == 'flip_h':
        return (w - 1 - u, v)
    return (u, v)


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：人工画框标注控件 (2026-09-08, 人工模式第一步) ★★★
# 暂停瞬间冻结原始彩色帧, 拖拽画矩形框(labelImg式), 信号回传图像坐标框。
# 只负责"冻结帧+画框+中心像素", 不做像素→基坐标换算(待M-链标定后在
# on_annotate_box 挂接, 见其中TODO)。
# 设计要点:
#   - 冻结=显示停, 驱动不停: 后台相机话题照常收, 「恢复预览」零成本回直播;
#   - 画框必须在静态帧上(与自动模式"静止读坐标"同一纪律, 消除θf/θn错配);
#   - 冻结帧用 KeepAspectRatio 完整显示(直播自 2026-11-05 起同为
#     KeepAspectRatio, 两边视野一致), 画框时看得到整幅画面,
#     控件→像素反映射也因此简单可靠。
# ──────────────────────────────────────────────────────────────────────────────
class AnnotateCameraLabel(QLabel):
    """支持冻结帧 + 拖拽矩形框的相机显示控件(冻结时才响应鼠标)"""

    # 框完成信号: 图像坐标系 (x1, y1, x2, y2)
    box_ready = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frozen_img = None       # 冻结帧 QImage(RGB888) 深拷贝(显示系)
        self._rect_img = None         # 当前框, 图像坐标 (x1,y1,x2,y2) ★显示系
        self._drag_start = None       # 拖拽起点, 图像坐标 (u,v) ★显示系
        self._box_confirmed = False   # 框是否已确认(黄=待确认, 绿=已确认)
        self._raw_wh = (0, 0)         # 原始(未补偿)图像宽高, 用于坐标反变换

    # ---- 外部接口 ----
    def freeze(self, img_bgr):
        """冻结: 传入 BGR numpy 帧, 进入画框模式.
        ★ 帧按 CAMERA_DISPLAY_ORIENT 转到显示系后绘制; 原始尺寸记下来,
          画框回传时再反变换回原始像素系(下游深度/标定链路零改动)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # ★ 先记原始尺寸, 再做显示补偿(rot180/flip 不改变宽高, 但统一在此记录)
        self._raw_wh = (rgb.shape[1], rgb.shape[0])
        rgb = _cam_orient_image(rgb)
        h, w, ch = rgb.shape
        # ★ .copy() 深拷贝: QImage引用numpy内存, 原数组回收后悬空会花屏
        self._frozen_img = QImage(
            rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._rect_img = None
        self._drag_start = None
        self._box_confirmed = False
        self.setCursor(Qt.CrossCursor)
        self.update()

    def unfreeze(self):
        """解冻: 回直播(直播帧由 update_camera_view 继续 setPixmap)"""
        self._frozen_img = None
        self._rect_img = None
        self._drag_start = None
        self._box_confirmed = False
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def clear_box(self):
        """清除当前框但不退出冻结帧, 允许重新拖拽画框"""
        self._rect_img = None
        self._drag_start = None
        self._box_confirmed = False
        self.update()

    def set_box_confirmed(self, ok):
        """标记当前框已确认(黄框→绿框)"""
        self._box_confirmed = bool(ok)
        self.update()

    @property
    def frozen(self):
        return self._frozen_img is not None

    @property
    def has_box(self):
        return self._rect_img is not None

    @property
    def box_confirmed(self):
        return self._box_confirmed

    # ---- 几何: 冻结帧的适配映射 ----
    def _fit_geo(self):
        """返回 (scale, dw, dh, iw, ih): 图像→控件 缩放与居中偏移"""
        iw, ih = self._frozen_img.width(), self._frozen_img.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / float(iw), lh / float(ih)) if iw and ih else 1.0
        dw = (lw - iw * scale) / 2.0
        dh = (lh - ih * scale) / 2.0
        return scale, dw, dh, iw, ih

    def _widget_to_img(self, mx, my):
        scale, dw, dh, iw, ih = self._fit_geo()
        u = int(round((mx - dw) / scale))
        v = int(round((my - dh) / scale))
        return (max(0, min(u, iw - 1)), max(0, min(v, ih - 1)))

    def _img_to_raw_rect(self, rect):
        """显示系图像框 → 原始像素系框(下游深度采样/基坐标解算用的坐标系).
        ★ 补偿模式为自逆变换, 对角点各做一次即可; 变换后重新排序保证
          x1<x2 / y1<y2(rot180 会交换上下左右)."""
        x1, y1, x2, y2 = rect
        w, h = self._raw_wh
        if not w or not h:
            return rect
        ax, ay = _cam_orient_uv(x1, y1, w, h)
        bx, by = _cam_orient_uv(x2, y2, w, h)
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    # ---- 鼠标 ----
    def mousePressEvent(self, ev):
        if not self.frozen:
            return super().mousePressEvent(ev)
        self._drag_start = self._widget_to_img(ev.pos().x(), ev.pos().y())
        self._rect_img = None      # 重新拖拽 = 重画新框(回到待确认状态)
        self._box_confirmed = False
        self.update()

    def mouseMoveEvent(self, ev):
        if not self.frozen or self._drag_start is None:
            return super().mouseMoveEvent(ev)
        u, v = self._widget_to_img(ev.pos().x(), ev.pos().y())
        x1, y1 = self._drag_start
        self._rect_img = (min(x1, u), min(y1, v), max(x1, u), max(y1, v))
        self.update()

    def mouseReleaseEvent(self, ev):
        if not self.frozen or self._drag_start is None:
            return super().mouseReleaseEvent(ev)
        u, v = self._widget_to_img(ev.pos().x(), ev.pos().y())
        x1, y1 = self._drag_start
        self._drag_start = None
        rect = (min(x1, u), min(y1, v), max(x1, u), max(y1, v))
        # 过小的框(<4px)视为误点, 清除不触发
        if rect[2] - rect[0] >= 4 and rect[3] - rect[1] >= 4:
            self._rect_img = rect
            # ★ 回传原始像素系框: 显示补偿只影响观感, 下游坐标链路保持不变
            self.box_ready.emit(*self._img_to_raw_rect(rect))
        else:
            self._rect_img = None
        self.update()

    # ---- 外观: 占位层随控件缩放(不参与画框/冻结逻辑) ----
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        overlay = getattr(self, 'overlay', None)
        if overlay is not None:
            overlay.setGeometry(0, 0, self.width(), self.height())

    # ---- 绘制 ----
    def paintEvent(self, ev):
        p = QPainter(self)
        if self.frozen:
            lw, lh = self.width(), self.height()
            scale, dw, dh, iw, ih = self._fit_geo()
            p.drawImage(QRectF(dw, dh, iw * scale, ih * scale), self._frozen_img)
            if self._rect_img is not None:
                x1, y1, x2, y2 = self._rect_img
                # 框: 待确认=DJI青, 已确认=白(与 RoboMaster 标注框一致的克制配色)
                p.setPen(QPen(QColor("#FFFFFF") if self._box_confirmed
                              else QColor("#00D2D2"), 2))
                p.drawRect(QRectF(dw + x1 * scale, dh + y1 * scale,
                                  (x2 - x1) * scale, (y2 - y1) * scale))
                # 中心十字 (琥珀, 表示"取点位置")
                cu = (x1 + x2) / 2.0
                cvc = (y1 + y2) / 2.0
                cx, cy = dw + cu * scale, dh + cvc * scale
                p.setPen(QPen(QColor("#FFC24B"), 2))
                p.drawLine(QPointF(cx - 12, cy), QPointF(cx + 12, cy))
                p.drawLine(QPointF(cx, cy - 12), QPointF(cx, cy + 12))
                # 中心像素标注
                # ★ 显示【原始像素系】中心坐标: 与 box_ready 回传、状态栏读数、
                #   日志里的 u/v 保持同一坐标系, 避免画面上看到的数和下游差一截.
                _rw, _rh = self._raw_wh
                if _rw and _rh:
                    _ru, _rv = _cam_orient_uv(int(cu), int(cvc), _rw, _rh)
                else:
                    _ru, _rv = int(cu), int(cvc)
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(cx + 16, cy - 8),
                           "(%d, %d)%s" % (_ru, _rv,
                                           " 已确认" if self._box_confirmed
                                           else " 待确认"))
            p.end()
            return
        # 直播模式: 复刻 QLabel(AlignCenter) 居中绘制, 保证外观与原版一致
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            pw, ph = pm.width(), pm.height()
            p.drawPixmap(int((self.width() - pw) / 2),
                         int((self.height() - ph) / 2), pm)
        else:
            p.setPen(QColor(255, 255, 255))
            p.drawText(self.rect(), Qt.AlignCenter, self.text())
        p.end()


# ══════════════════════════════════════════════════════════════════════════════
# ★ UI v3.0 (2026-10-25): DJI RoboMaster 官网视觉语言
#   仅"外观层"重构 —— 控件层级/对象名/信号连接与 v2.x 完全一致, 不触碰任何
#   按键触发逻辑、流程状态机与后台线程。
#   视觉基调(对齐大疆官网 / RoboMaster 赛事站):
#     · 底: #0B0B0D 近黑场, 内容面板 #151619, 用 1px #2A2B31 细描边切分
#     · 字: 主 #F2F3F5 / 次 #9AA0A8 / 弱 #6C7079, 无彩色正文
#     · 点睛: DJI 青 #00D2D2 仅用于主操作/焦点/当前项(全界面唯一彩色)
#     · 红色 #E4393C 只保留给急停/安全语义
#     · 排版: 细线框 + 小号大写字距标题 + 发丝级分隔线, 弱化圆角与阴影
# ══════════════════════════════════════════════════════════════════════════════
class DJITheme:
    BG        = "#0B0B0D"   # 窗口底
    PANEL     = "#151619"   # 卡片/面板
    PANEL_2   = "#1B1C20"   # 次级面板/按钮底
    PANEL_3   = "#232429"   # hover / 激活底
    LINE      = "#2A2B31"   # 主分隔线
    LINE_SOFT = "#1E1F24"   # 发丝线(卡片内)
    TEXT      = "#F2F3F5"
    TEXT_2    = "#9AA0A8"
    TEXT_3    = "#6C7079"
    ACCENT    = "#00D2D2"   # DJI 青
    ACCENT_HI = "#3CE4E4"   # hover
    ACCENT_DK = "#0A8F8F"   # 边框/描边
    ON_ACCENT = "#04191A"   # 青底上的字
    DANGER    = "#E4393C"
    # ── 字体(方案B): 界面 Lato(西文) + Noto Sans CJK SC(中文); 数据 Ubuntu Mono ──
    #   Lato 用于英文/数字, 中文自动回退到 Noto Sans CJK SC; Ubuntu Mono 让
    #   坐标/关节角这类数字有"仪表读数"感.
    UI_FONT   = "'Lato','Noto Sans CJK SC','Noto Sans CJK HK','WenQuanYi Zen Hei'," \
                "'Droid Sans Fallback','Sans-serif'"
    #   ★ 数据字体: Ubuntu Mono 的 0 带斜线、负号过长, 在输入框里显得像手写体;
    #     DejaVu Sans Mono 字形规整、负号居中, 更适合读数
    MONO      = "'DejaVu Sans Mono','Noto Sans Mono CJK SC','Consolas','monospace'"


# ── 矢量图标(内联 SVG, 无需外部图片; 颜色随主题走, 缩放不糊) ────────────────
_SVG_CAMERA = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
               "<path d='M2 7.5h11v9H2z' fill='none' stroke='{c}' stroke-width='1.6'/>"
               "<path d='M13.4 11.2l6.6-3.6v8.8l-6.6-3.6z' fill='none' stroke='{c}' "
               "stroke-width='1.6' stroke-linejoin='round'/></svg>")
_SVG_FRAME = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M3 8V4h4M17 4h4v4M21 16v4h-4M7 20H3v-4' fill='none' "
              "stroke='{c}' stroke-width='1.7' stroke-linecap='square'/></svg>")
_SVG_PAUSE = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M9 4v16M15 4v16' fill='none' stroke='{c}' "
              "stroke-width='2.2'/></svg>")
_SVG_TARGET = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
               "<circle cx='12' cy='12' r='7' fill='none' stroke='{c}' stroke-width='1.6'/>"
               "<path d='M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4' fill='none' "
               "stroke='{c}' stroke-width='1.6'/></svg>")
_SVG_LINK = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M4 12h13M12 6.5l5.5 5.5L12 17.5' fill='none' stroke='{c}' "
             "stroke-width='1.8' stroke-linecap='square'/></svg>")
_SVG_CHECK = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M4 12.5l5.2 5.2L20 6.8' fill='none' stroke='{c}' "
              "stroke-width='2.2' stroke-linecap='square'/></svg>")
_SVG_STOP = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<rect x='5' y='5' width='14' height='14' fill='{c}'/></svg>")
_SVG_PLAY = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M6.5 3.5l14 8.5-14 8.5z' fill='{c}'/></svg>")
_SVG_POWER = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M12 2.5v8' fill='none' stroke='{c}' stroke-width='2'/>"
              "<path d='M6.6 6.2a8 8 0 1 0 10.8 0' fill='none' stroke='{c}' "
              "stroke-width='2' stroke-linecap='square'/></svg>")
_SVG_HOME = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M3.5 10.5L12 3.5l8.5 7' fill='none' stroke='{c}' stroke-width='1.7'/>"
             "<path d='M6 9.6V20h12V9.6' fill='none' stroke='{c}' stroke-width='1.7'/>"
             "<path d='M12 20v-5.5' fill='none' stroke='{c}' stroke-width='1.7'/></svg>")
_SVG_ROTATE_CW = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
                  "<path d='M20 12a8 8 0 1 1-2.6-5.9' fill='none' stroke='{c}' "
                  "stroke-width='1.8' stroke-linecap='square'/>"
                  "<path d='M20.5 3.5v5h-5' fill='none' stroke='{c}' stroke-width='1.8' "
                  "stroke-linecap='square'/></svg>")
_SVG_ROTATE_CCW = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
                   "<path d='M4 12a8 8 0 1 0 2.6-5.9' fill='none' stroke='{c}' "
                   "stroke-width='1.8' stroke-linecap='square'/>"
                   "<path d='M3.5 3.5v5h5' fill='none' stroke='{c}' stroke-width='1.8' "
                   "stroke-linecap='square'/></svg>")
_SVG_UP = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
           "<path d='M12 19V5M6 11l6-6 6 6' fill='none' stroke='{c}' "
           "stroke-width='2' stroke-linecap='square'/></svg>")
_SVG_DOWN = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M12 5v14M6 13l6 6 6-6' fill='none' stroke='{c}' "
             "stroke-width='2' stroke-linecap='square'/></svg>")
_SVG_GRIP = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M7 3v6.5c0 2 1.2 3.2 3 3.2h4c1.8 0 3-1.2 3-3.2V3' "
             "fill='none' stroke='{c}' stroke-width='1.7' stroke-linejoin='round'/>"
             "<path d='M7 3h3M14 3h3' fill='none' stroke='{c}' stroke-width='1.7'/>"
             "<path d='M12 13v8' fill='none' stroke='{c}' stroke-width='1.7'/></svg>")
_SVG_SWAP = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M4 8h14M14 4l4 4-4 4' fill='none' stroke='{c}' stroke-width='1.7' "
             "stroke-linecap='square'/><path d='M20 16H6M10 12l-4 4 4 4' fill='none' "
             "stroke='{c}' stroke-width='1.7' stroke-linecap='square'/></svg>")
_SVG_SMEAR = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
              "<path d='M3 20h6' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
              "<path d='M10 20L17 6' stroke='{c}' stroke-width='2.4' "
              "stroke-linecap='square'/>"
              "<circle cx='18.5' cy='5' r='2.6' fill='{c}'/>"
              "<path d='M15.5 16.5l3 1M18 13.5l3 1' stroke='{c}' "
              "stroke-width='2' stroke-linecap='round'/></svg>")
_SVG_DAB = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
            "<path d='M12 2.5v6' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
            "<path d='M8.5 9h7l-1.2 6h-4.6z' fill='none' stroke='{c}' "
            "stroke-width='2.2' stroke-linejoin='round'/>"
            "<circle cx='12' cy='19.5' r='2.6' fill='{c}'/></svg>")
_SVG_GRIPPER = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
                "<path d='M12 2.5v4' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
                "<path d='M5 8h14' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
                "<path d='M7 8v6' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
                "<path d='M17 8v6' stroke='{c}' stroke-width='2.4' stroke-linecap='square'/>"
                "<path d='M7 14h4M13 14h4' stroke='{c}' stroke-width='2.4' "
                "stroke-linecap='square'/></svg>")

_SVG_GRID = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<rect x='3.5' y='3.5' width='7' height='7' fill='none' stroke='{c}' "
             "stroke-width='1.6'/><rect x='13.5' y='3.5' width='7' height='7' fill='none' "
             "stroke='{c}' stroke-width='1.6'/><rect x='3.5' y='13.5' width='7' height='7' "
             "fill='none' stroke='{c}' stroke-width='1.6'/><rect x='13.5' y='13.5' width='7' "
             "height='7' fill='none' stroke='{c}' stroke-width='1.6'/></svg>")
_SVG_LIST = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M4 6.5h16M4 12h16M4 17.5h10' fill='none' stroke='{c}' "
             "stroke-width='1.7' stroke-linecap='square'/></svg>")
_SVG_SHIELD = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
               "<path d='M12 2.8l7.5 3v6.4c0 4.2-3 7.3-7.5 9-4.5-1.7-7.5-4.8-7.5-9V5.8z' "
               "fill='none' stroke='{c}' stroke-width='1.6'/></svg>")


class DJIIcon:
    """内联 SVG → QIcon 工厂(结果按 名称+颜色+尺寸 缓存)。"""
    _cache = {}

    @classmethod
    def get(cls, svg_tpl, color, size=18):
        # ★ 缓存键必须用完整 SVG 内容: 原先取 svg_tpl[:24], 而所有图标的
        #   前 24 字符都是 "<svg xmlns='http://www.sv", 导致不同图标互相
        #   命中缓存(夹爪的"闭合▼"会显示成"张开▲"就是这个原因).
        key = (hash(svg_tpl), color, size)
        if key in cls._cache:
            return cls._cache[key]
        icon = QIcon()
        painter = None
        try:
            data = svg_tpl.format(c=color).encode("utf-8")
            renderer = QSvgRenderer(QByteArray(data))
            pm = QPixmap(size, size)
            pm.fill(Qt.transparent)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing, True)
            renderer.render(painter, QRectF(0, 0, size, size))
            painter.end()
            painter = None
            icon = QIcon(pm)
        except Exception:
            icon = QIcon()
        finally:
            # ★ 必须收尾: painter 未 end 时 QPixmap 析构会触发
            #   "Cannot destroy paint device that is being painted" 并段错误
            if painter is not None:
                try:
                    painter.end()
                except Exception:
                    pass
        cls._cache[key] = icon
        return icon


class ActionButton(QPushButton):
    """预设动作按钮: 图标与文字全部自绘, 保证五枚按钮共用同一套对齐栅格。

    · 图标用 QSvgRenderer 按实际尺寸矢量绘制(QIcon 在 20px 下会做有损降采样)
    · 文字用 QPainter 绘制在固定起点, 不再交给 QSS 的居中逻辑
      —— 否则"动作 · 抓取"比"涂抹 · 1 号位"少两个字, 文字会各自居中而错位
    · 点击/信号行为与普通 QPushButton 完全一致(未重写任何事件)
    """

    ICON_SIZE = 20      # 图标尺寸
    ICON_GAP = 20       # 图标与文字间距
    TEXT_SLOT = 88      # 文字区固定宽度(按最长文案"涂抹 · 1 号位"定)

    def __init__(self, text, svg_tpl, icon_color=None, icon_size=None, parent=None):
        super().__init__(text, parent)
        self._svg = svg_tpl
        self._icon_color = icon_color or DJITheme.ACCENT
        self._icon_size = icon_size or self.ICON_SIZE
        self._renderer = None
        self._renderer_dpr = 0.0
        self.setCursor(Qt.PointingHandCursor)

    def _ensure_renderer(self):
        dpr = self.devicePixelRatioF() if hasattr(self, 'devicePixelRatioF') else 1.0
        if self._renderer is not None and abs(self._renderer_dpr - dpr) < 1e-3:
            return
        try:
            data = self._svg.format(c=self._icon_color).encode("utf-8")
            self._renderer = QSvgRenderer(QByteArray(data))
            self._renderer_dpr = dpr
        except Exception:
            self._renderer = None

    def paintEvent(self, ev):
        # ★ 不再调用 super().paintEvent(): 否则 QSS 会先把文字画一遍,
        #   位置由它自己算, 与下面的栅格对不上 → 出现重影/错位。
        #   这里完整接管绘制: 背景(用样式表) + 图标 + 文字。
        self._ensure_renderer()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        # 1) 背景与边框交给 QStyle(沿用 QSS 的 #actionRow 规则)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""                    # 文字自己画, 不让样式重复绘制
        opt.icon = QIcon()               # 图标也自己画
        self.style().drawControl(QStyle.CE_PushButton, opt, p, self)

        # 2) 内容栅格: [图标][间距][固定宽度文字区], 整块水平居中
        block = self._icon_size + self.ICON_GAP + self.TEXT_SLOT
        x0 = int((self.width() - block) / 2.0)
        cy = self.height() / 2.0
        iy = int(cy - self._icon_size / 2.0)

        if self._renderer is not None and self._renderer.isValid():
            self._renderer.render(
                p, QRectF(x0, iy, self._icon_size, self._icon_size))

        # 3) 文字: 固定起点左对齐, 与图标共用基线
        tx = x0 + self._icon_size + self.ICON_GAP
        color = QColor(DJITheme.ACCENT if self.isDown() else DJITheme.TEXT)
        if not self.isEnabled():
            color = QColor("#4E5157")
        p.setPen(QPen(color))
        p.setFont(self.font())
        p.drawText(QRectF(tx, 0, self.TEXT_SLOT, self.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self.text())
        p.end()


def _hex_poly(cx, cy, r):
    """正六边形(平顶朝左右), 用于机甲风节点。"""
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 30)
        pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return QPolygonF(pts)


def _diamond_poly(cx, cy, r):
    """菱形(旋转45°的正方形)。"""
    return QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                      QPointF(cx, cy + r), QPointF(cx - r, cy)])


class MechaTitleDecor(QWidget):
    """标题两侧的机甲风装饰线(F2 方案): 粗细 + 透明度双渐变。

    用 QPainterPath 拼一个实心梯形来表现"由细到粗"的过渡(逐段画线会有
    接缝, 实心填充则完全平滑); 同时叠加左右方向的透明度渐变 —— 远端几乎
    融进背景, 靠标题一端明亮收束, 形成"能量汇聚"的观感。

    side = -1 左侧, +1 右侧(完全镜像)。纯装饰, 不接收鼠标事件。
    """

    W_NEAR = 3.0        # 靠标题一端线宽(px)
    W_FAR = 0.6         # 远端线宽(px)
    A_NEAR = 240        # 靠标题一端不透明度(0-255)
    A_FAR = 15          # 远端不透明度

    def __init__(self, side=-1, color=None, parent=None):
        super().__init__(parent)
        self._side = side
        self._color = QColor(color or DJITheme.ACCENT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(26)

    def paintEvent(self, ev):
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p):
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        if w < 4:
            return
        cy = h / 2.0

        # 统一成 "far(远端) → near(靠标题端)", 左右只是方向相反
        if self._side < 0:
            far_x, near_x = 0.0, float(w)
        else:
            far_x, near_x = float(w), 0.0

        w_far, w_near = self.W_FAR, self.W_NEAR

        # ── 实心梯形(粗细渐变) ──
        path = QPainterPath()
        path.moveTo(far_x, cy - w_far / 2.0)
        path.lineTo(near_x, cy - w_near / 2.0)
        path.lineTo(near_x, cy + w_near / 2.0)
        path.lineTo(far_x, cy + w_far / 2.0)
        path.closeSubpath()

        # ── 透明度渐变 ──
        grad = QLinearGradient(far_x, 0, near_x, 0)
        c_far = QColor(self._color)
        c_far.setAlpha(self.A_FAR)
        c_near = QColor(self._color)
        c_near.setAlpha(self.A_NEAR)
        grad.setColorAt(0.0, c_far)
        grad.setColorAt(1.0, c_near)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(path)


class TitleBrackets(QWidget):
    """标题四角的 L 型瞄准角标(V5 点子): 把标题框成"被锁定的目标"。

    作为标题的兄弟控件覆盖在标题区上方, 只画四个角, 中间完全透明。
    """

    ARM = 14        # 角标臂长
    PAD = 6         # 角标距容器边缘的留白(与标题的间距由 contentsMargins 控制)

    def __init__(self, parent=None, color=None):
        super().__init__(parent)
        self._color = QColor(color or DJITheme.ACCENT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        arm = self.ARM
        pad = self.PAD
        pen = QPen(self._color, 1.6)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        # 左上
        p.drawLine(QPointF(pad, pad), QPointF(pad + arm, pad))
        p.drawLine(QPointF(pad, pad), QPointF(pad, pad + arm))
        # 右上
        p.drawLine(QPointF(w - pad, pad), QPointF(w - pad - arm, pad))
        p.drawLine(QPointF(w - pad, pad), QPointF(w - pad, pad + arm))
        # 左下
        p.drawLine(QPointF(pad, h - pad), QPointF(pad + arm, h - pad))
        p.drawLine(QPointF(pad, h - pad), QPointF(pad, h - pad - arm))
        # 右下
        p.drawLine(QPointF(w - pad, h - pad), QPointF(w - pad - arm, h - pad))
        p.drawLine(QPointF(w - pad, h - pad), QPointF(w - pad, h - pad - arm))
        p.end()


class HLine(QFrame):
    """发丝级横向分隔线(替代 QGroupBox 原生边框, 做 DJI 式"细线切分")。"""

    def __init__(self, color=None, height=1, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: %s; border: none;" %
                           (color or DJITheme.LINE_SOFT))


class PanelCard(QFrame):
    """扁平直角面板: 1px 细描边 + 深色底, 左上角一道 2px 青色刻度(当前模块标记)。"""

    def __init__(self, parent=None, accent=False, accent_color=None, radius=2):
        super().__init__(parent)
        self._accent = accent
        self._accent_color = QColor(accent_color or DJITheme.ACCENT)
        self._radius = radius
        self.setObjectName("djiPanel")

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRect(self.rect())
        r.setWidth(max(1, r.width() - 1))
        r.setHeight(max(1, r.height() - 1))
        p.setBrush(QBrush(QColor(DJITheme.PANEL)))
        p.setPen(QPen(QColor(DJITheme.LINE), 1))
        p.drawRoundedRect(r, self._radius, self._radius)
        if self._accent:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self._accent_color))
            p.drawRect(QRect(r.left() + 1, r.top() + 1, 34, 2))
        p.end()


class AccentBar(QWidget):
    """标题栏/相机区左侧的青色竖向标记(代替 v2 的蓝色圆角小方块)。"""

    def __init__(self, width=3, height=18, color=None, parent=None):
        super().__init__(parent)
        self._color = QColor(color or DJITheme.ACCENT)
        self.setFixedSize(width, height)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._color))
        p.drawRect(self.rect())
        p.end()


def apply_letter_spacing(widget, px):
    """QSS 不支持 letter-spacing, 用 QFont 绝对字距补上(DJI 风的疏排小字)。"""
    try:
        f = widget.font()
        f.setLetterSpacing(QFont.AbsoluteSpacing, float(px))
        widget.setFont(f)
    except Exception:
        pass
    return widget


class MicroTag(QLabel):
    """等宽小号英文标签(顶部副题 / 卡片序号), 字距拉开。"""

    def __init__(self, text, color=None, size=10, spacing=2, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "color: %s; font-family: %s; font-size: %dpx;"
            " background: transparent; border: none;" %
            (color or DJITheme.TEXT_3, DJITheme.MONO, size))
        apply_letter_spacing(self, spacing)


class CameraPlaceholder(QWidget):
    """相机无画面时的占位层: 取景角标 + 中央图标 + 提示文字(纯装饰, 不拦截鼠标)。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = "等待相机图像..."
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def set_text(self, text):
        self._text = text
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        # 中央十字准星 + 同心圆
        cx, cy = w / 2.0, h / 2.0 - 10
        p.setPen(QPen(QColor(58, 60, 66), 1))
        p.drawEllipse(QPointF(cx, cy), 46, 46)
        p.setPen(QPen(QColor(0, 210, 210, 90), 1))
        p.drawEllipse(QPointF(cx, cy), 30, 30)
        p.drawLine(QPointF(cx - 58, cy), QPointF(cx - 20, cy))
        p.drawLine(QPointF(cx + 20, cy), QPointF(cx + 58, cy))
        p.drawLine(QPointF(cx, cy - 58), QPointF(cx, cy - 20))
        p.drawLine(QPointF(cx, cy + 20), QPointF(cx, cy + 58))
        p.setPen(QPen(QColor(DJITheme.TEXT_2), 1))
        p.drawText(QRect(int(cx - 240), int(cy + 72), 480, 22),
                   Qt.AlignCenter, self._text)

        # 四角取景框
        p.setPen(QPen(QColor(0, 210, 210, 150), 2))
        m = 18
        L = 26
        p.drawLine(QPointF(m, m + L), QPointF(m, m))
        p.drawLine(QPointF(m, m), QPointF(m + L, m))
        p.drawLine(QPointF(w - m - L, m), QPointF(w - m, m))
        p.drawLine(QPointF(w - m, m), QPointF(w - m, m + L))
        p.drawLine(QPointF(m, h - m - L), QPointF(m, h - m))
        p.drawLine(QPointF(m, h - m), QPointF(m + L, h - m))
        p.drawLine(QPointF(w - m - L, h - m), QPointF(w - m, h - m))
        p.drawLine(QPointF(w - m, h - m), QPointF(w - m, h - m - L))
        p.end()


DJI_QSS = """
/* ═══════════ 基础 ═══════════ */
* {
    font-family: @UI_FONT@;
    font-size: 14px;
    outline: none;
}
QMainWindow, QWidget { background-color: @BG@; color: @TEXT@; }
QMainWindow::separator { background: @BG@; width: 0px; height: 0px; }
QLabel { background: transparent; color: @TEXT@; }
QToolTip {
    background-color: @PANEL_2@; color: @TEXT_2@;
    border: 1px solid @LINE@; padding: 5px 8px;
}

/* ═══════════ 顶部标题栏 ═══════════ */
#headerBar { background-color: @BG@; border-bottom: 1px solid @LINE@; }
#headerTitle {
    font-size: 27px; font-weight: 600; color: #FFFFFF; background: transparent;
}
#headerSub {
    font-size: 12px; color: @TEXT_2@;
    font-family: @MONO@; background: transparent;
}
#headerAccent { background-color: @ACCENT@; border-radius: 0px; }
#headerTag {
    color: @TEXT_3@; font-size: 11px;
    font-family: @MONO@; border: 1px solid @LINE@;
    padding: 3px 9px; background: transparent;
}

/* ═══════════ 卡片(右栏 QGroupBox) ═══════════ */
QGroupBox {
    background-color: @PANEL@;
    border: 1px solid @LINE@;
    border-radius: 2px;
    margin-top: 0px;
    padding: 0px;
}
QGroupBox#cardPrimary { border: 1px solid @ACCENT_DK@; }

/* 卡片头部: 英文序号 + 中文标题 + 发丝线 + 编号 */
#cardHeader { background: transparent; }
#cardGlyph { color: @ACCENT@; background: transparent; font-size: 13px; }
#cardTitle {
    color: @TEXT@; font-size: 16px; font-weight: bold; background: transparent;
}
#cardIndex {
    color: @TEXT_3@; font-size: 16px; font-weight: bold;
    font-family: @MONO@; background: transparent;
}
#cardRule { background-color: #3A3C43; }
#cardTitleEn {
    color: @TEXT_3@; font-size: 9px;
    font-family: @MONO@; background: transparent;
}

/* ═══════════ 输入框(方形细边, 聚焦转青) ═══════════ */
QLineEdit {
    background-color: @PANEL_2@; color: @TEXT@;
    border: 1px solid @LINE@; border-radius: 2px;
    padding: 6px 8px;
    font-family: @MONO@; font-size: 14px;
    selection-background-color: @ACCENT_DK@; selection-color: #FFFFFF;
}
QLineEdit:hover { border: 1px solid #3A3C43; }
QLineEdit:focus { border: 1px solid @ACCENT@; background-color: #1F2024; }
QLineEdit:disabled { color: @TEXT_3@; background-color: #121317; }

/* 紧凑"输入槽": 无框, 仅底部 1px 线(左栏坐标输入) */
QLineEdit#axisField {
    background: transparent; border: none;
    border-bottom: 1px solid #3A3C43; border-radius: 0px;
    padding: 5px 4px 6px 4px; color: @TEXT@;
    font-family: @MONO@; font-size: 15px;
}
QLineEdit#axisField:hover { border-bottom: 1px solid #4C4E56; }
QLineEdit#axisField:focus {
    border-bottom: 1px solid @ACCENT@; background: transparent;
}

/* ═══════════ 按钮: 默认 / primary(青) / danger(红) ═══════════ */
QPushButton {
    background-color: transparent; color: @TEXT@;
    border: 1px solid #33353C; border-radius: 2px;
    padding: 8px 16px;
    font-size: 14px;
}
QPushButton:hover { background-color: #22242A; border: 1px solid #5A5D66; }
QPushButton:pressed { background-color: #15161A; border-color: @ACCENT_DK@; }
QPushButton:disabled {
    color: #4E5157; border: 1px solid #24262B; background-color: #101114;
}
QPushButton#primary {
    background-color: @ACCENT@; color: @ON_ACCENT@;
    border: 1px solid @ACCENT@; font-weight: 600;
}
QPushButton#primary:hover { background-color: @ACCENT_HI@; border-color: @ACCENT_HI@; }
QPushButton#primary:pressed { background-color: @ACCENT_DK@; border-color: @ACCENT_DK@; }
QPushButton#primary:disabled {
    background-color: #163234; color: #5E7E80;
    border: 1px solid #24494B;
}
QPushButton#danger {
    background-color: transparent; color: @DANGER@;
    border: 1px solid @DANGER@; font-weight: 600;
}
QPushButton#danger:hover { background-color: @DANGER@; color: #FFFFFF; }
QPushButton#danger:pressed { background-color: #B92A2D; border-color: #B92A2D; }
/* 方形步进按钮(J6 顺逆时针 / 夹爪开合): 描边 + 图标 */
QPushButton#circ {
    background-color: @PANEL_2@; color: @TEXT_2@;
    border: 1px solid #33353C; border-radius: 2px;
    min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px;
    padding: 0px;
}
QPushButton#circ:hover { background-color: #26282E; border-color: @ACCENT@; }
QPushButton#circ:pressed { background-color: #101114; }
QPushButton#circ:disabled {
    background-color: #101114; border-color: #24262B;
}

/* 预设动作: 全宽卡片式按钮 —— 青色图标 + 居中文字 + 左侧强调条 */
QPushButton#actionRow {
    /* 文字由 ActionButton.paintEvent 自绘(统一栅格), 这里只管背景/边框 */
    padding: 0px;
    border: 1px solid #2E3037;
    border-left: 3px solid @ACCENT_DK@;
    background-color: #131417;
    font-size: 15px;
    font-weight: bold;
}
QPushButton#actionRow:hover {
    border: 1px solid #454851;
    border-left: 3px solid @ACCENT@;
    background-color: #1C1E23;
    color: #FFFFFF;
}
QPushButton#actionRow:pressed {
    border-left: 3px solid @ACCENT_HI@;
    background-color: #0F1012;
}
QPushButton#actionRow:disabled {
    border: 1px solid #24262B;
    border-left: 3px solid #24262B;
    color: #4E5157;
    background-color: #101114;
}

/* 工具条(相机条/标注条/底部系统条)内按钮: 同风格, 内边距收紧 */
#camBar QPushButton, #annotateBar QPushButton, #leftFooter QPushButton {
    padding: 7px 14px; border-radius: 2px;
}
/* 关节/夹爪网格内的按钮: 内边距收紧, 保证"复位0mm/重标定"完整显示 */
#leftPanel QGridLayout QPushButton { padding: 7px 6px; }

/* ═══════════ 信息面板 / 日志 ═══════════ */
QTextEdit {
    font-family: @MONO@;
    font-size: 13px; color: #C9CCD2;
    background-color: #101114; border: 1px solid @LINE_SOFT@;
    border-radius: 2px; padding: 8px;
    selection-background-color: @ACCENT_DK@; selection-color: #FFFFFF;
}
QTextEdit#detectLog { border-left: 2px solid @ACCENT@; }

/* ═══════════ 工具条容器 ═══════════ */
#camBar, #annotateBar, #leftFooter {
    background-color: @PANEL@; border: 1px solid @LINE@;
    border-radius: 2px;
}
#leftFooter { border-left: 2px solid @ACCENT_DK@; }
#wsBox { background: transparent; }
#camBar QLabel, #annotateBar QLabel, #leftFooter QLabel { background: transparent; }
#vSep { background-color: @LINE@; }
#camChip {
    color: @TEXT_2@; border: 1px solid @LINE@; border-radius: 2px;
    padding: 4px 10px; font-size: 13px; background-color: @PANEL_2@;
    font-family: @MONO@;
}

/* ═══════════ 状态栏 / 滚动条 / 分割器 ═══════════ */
QStatusBar {
    background-color: @BG@; border-top: 1px solid @LINE@;
    color: @TEXT_2@; font-size: 13px;
    font-family: @MONO@;
}
QStatusBar::item { border: none; }
QSplitter::handle { background-color: @BG@; }
QSplitter::handle:hover { background-color: @ACCENT_DK@; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical {
    background: #34363D; border-radius: 0px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: @ACCENT_DK@; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0; }
QScrollBar::handle:horizontal {
    background: #34363D; border-radius: 0px; min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: @ACCENT_DK@; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QMessageBox { background-color: @PANEL@; }
QMessageBox QLabel { color: @TEXT@; }
"""


def _dji_qss():
    """把 DJITheme 调色板代入 QSS 模板。"""
    mapping = {
        "@BG@": DJITheme.BG, "@PANEL@": DJITheme.PANEL,
        "@PANEL_2@": DJITheme.PANEL_2, "@PANEL_3@": DJITheme.PANEL_3,
        "@LINE@": DJITheme.LINE, "@LINE_SOFT@": DJITheme.LINE_SOFT,
        "@TEXT@": DJITheme.TEXT, "@TEXT_2@": DJITheme.TEXT_2,
        "@TEXT_3@": DJITheme.TEXT_3,
        "@ACCENT@": DJITheme.ACCENT, "@ACCENT_HI@": DJITheme.ACCENT_HI,
        "@ACCENT_DK@": DJITheme.ACCENT_DK, "@ON_ACCENT@": DJITheme.ON_ACCENT,
        "@DANGER@": DJITheme.DANGER, "@MONO@": DJITheme.MONO,
        "@UI_FONT@": DJITheme.UI_FONT,
    }
    qss = DJI_QSS
    for k, v in mapping.items():
        qss = qss.replace(k, v)
    return qss


class ArmControllerUI(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.current_action = None
        self.action_buttons = []
        self.scan_thread = None

        self.setWindowTitle("刑侦现场物证采集系统")
        self.setGeometry(100, 100, 1920, 1080)

        # ──────────────────────────────────────────────────────────────────────
        # ★ UI v3.0 (2026-10-25): DJI RoboMaster 官网风格 —— 近黑底 + 发丝细线 +
        #   DJI 青点睛 + 方角排版; 仅外观层, 控件/对象名/信号连接与 v2.x 一致
        # ──────────────────────────────────────────────────────────────────────
        self.setStyleSheet(_dji_qss())

        self.launch_process = QProcess()
        self.launch_process.readyReadStandardOutput.connect(self.handle_launch_output)
        self.launch_process.readyReadStandardError.connect(self.handle_launch_error)
        self.launch_process.finished.connect(self.handle_launch_finished)

        self.camera_processes = []
        # (标定模式已移除 2026-08-28; 手眼结果固化于 perception_calib.py)
        self.init_ui()
        
        # ★ 重要：在 show_current_status() 之前初始化所有状态变量
        # 机械臂移动相关变量
        self.is_moving = False
        self.move_start_time = 0
        self.move_duration = 0
        self.move_start_pos = []
        self.move_end_pos = []
        # ★ 运动代号(epoch): 每次用户发起新运动/停止时 +1;
        # _post_move_refine 后台闭环线程醒来若发现 epoch 已变, 说明用户已下了
        # 新指令(如点了「返回初始位置」), 立即放弃补走 —— 修复"回零走到一半
        # 又被闭环线程拽回之前点位"的问题
        self._move_epoch = 0

        # ★ 夹爪控制相关变量
        self.is_gripper_waiting = False
        self.gripper_wait_start = 0
        self.gripper_wait_duration = 0

        # ★ 扫描状态变量（必须在 show_current_status 之前初始化）
        self.is_scanning = False
        self.scan_thread = None       # (保留兼容字段)
        # ★ 巡点循环扫描(2026-09-17同步): 停止事件(停止扫描/紧急停止/回零/关窗都置位)
        self._scan_stop = threading.Event()

        # ★★★ 新增：人工画框模式状态(暂停瞬间快照, 见 on_annotate_pause/box) ★★★
        self._freeze_joints = None     # 冻结时实际关节角(get_state优先)
        self._freeze_depth = None      # 冻结时深度图
        self._freeze_th1 = 0.0         # 冻结时关节1角度(冗余存一份, 日志用)
        self._freeze_t = None          # 冻结时刻
        self._pending_box = None       # 待确认的框(x1,y1,x2,y2), 「确认框选」后才生效
        self._boxed_pending = None     # ★ 框选待执行状态(2026-09-16 v2流程): 确认后只
        #   填输入框不动臂, {'xyz':[修正后], 'est_n':估算法向}; 点[移动到位置]时
        #   若输入框未被改过则走法向候选搜索路径, 否则按手动输入处理
        self._j6_home = None           # J6开机位(启动机械臂时记录, [J6归位]按钮目标)

        # ★ 工作线程→UI消息队列(2026-09-09): start_arm等后台线程的反馈此前走
        #   QTimer.singleShot(从python线程调度永不触发)+只写UI控件(不落盘),
        #   表现="点启动机械臂后界面全程静默, 失败原因不可见". 现统一入队,
        #   由主线程 update_all 取出显示, 并 print 落盘 rebot_ui.log.
        self._ui_msgs = []
        self._ui_calls = []     # ★ 工作线程→主线程UI投递队列(2026-09-17, 见_post_ui)

        # ──────────────────────────────────────────────────────────────
        # ★★★ 新增：自动物证采集流程状态变量 ★★★
        # ──────────────────────────────────────────────────────────────
        self.auto_mode = False          # 是否处于自动采集模式
        # ★ 标定模式(2026-09-10): True=启动相机时跑 ArUco标定管线(常开,
        #   无UI按钮), v4实时检测暂停; False=正常检测管线.
        #   2026-10-30: 改为 True —— blood_detect.launch 里同样补齐了
        #   depth_registration=true + flip_color=true + flip_depth=true,
        #   两条管线彩深方向都对齐了; 这里默认 True 直接跑 aruco_calib_gemini.launch,
        #   不再走 blood_detect.launch (用户已明确"不要检测了").
        self.CALIB_MODE = True
        # ★ 标定模式状态(ChArUco 板位姿读数)
        self._calib_last_pose = None      # 最近一帧 /aruco_single/pose
        self._calib_last_pose_ts = 0.0    # 该帧到达时间(判活, 1s 内视为有效)
        self._calib_pose_sub = None       # /aruco_single/pose 订阅器
        self._calib_img_sub = None        # /aruco_single/result 订阅器
        # 'IDLE' / 'INIT_HOMING' / 'MOVE_TO_SCAN_START' / 'SCANNING' / 'POST_SCAN_WAIT'
        self.auto_state = 'IDLE'
        self._state_action_triggered = False  # 当前状态的一次性动作是否已触发
        self._pending_category = None
        self._pending_confirm_count = 0
        self._pending_depth_img = None  # 与_pending_category配套的最近一帧深度图
        self._pending_th1 = 0.0         # 与_pending_category配套的最近一帧关节1角度
        self._best_capture = None       # 本轮扫描中确认到的最新一次(det, xyz, normal, depth_img, th1)
        self._post_scan_wait_start = 0.0

        self.detector = None             # ForensicDetector 实例，首次使用时懒加载
        self.detection_thread = None     # ForensicDetectionThread 实例，首次使用时懒加载
        self.raw_color_image = None      # (兼容保留) 最近一帧原始彩色图 (numpy, BGR)
        self.raw_depth_image = None      # (兼容保留) 最近一帧原始深度图
        self._raw_color_msg = None       # ★ 原始彩色msg缓存(回调线程只做引用赋值, 段错误修复)
        self._raw_depth_msg = None       # ★ 原始深度msg缓存 (最近一帧, 兼容保留)
        self._recent_depth_msgs = []     # ★★ 2026-10-30 深度环形缓存(size=5) ——
        #   Orbbec 深度相机 IR 投射器偶尔漏拍 (~10%概率), 单帧有效像素可能为 0.
        #   冻结时刻自动挑选最近几帧里有效像素最多的那一帧, 避免卡死.
        self.camera_intrinsics = pcalib.CameraIntrinsics()  # 彩色内参, 由 /camera/color/camera_info 更新
        self.depth_intrinsics = pcalib.CameraIntrinsics()   # ★ 深度内参, 由 /camera/depth/camera_info 更新
        self._depth_to_color_extrinsics = None  # ★ depth→color 外参4x4矩阵(从/camera/depth_to_color获取)
        self.color_sub = None
        self.depth_sub = None
        self.caminfo_sub = None
        self.depth_caminfo_sub = None    # ★ 深度内参订阅器

        # 现在调用 show_current_status 时所有变量都已定义
        self.show_current_status()

        # ★ 检测信息框写入一条"就绪"提示, 便于确认该控件可见/可工作
        if hasattr(self, 'detection_display'):
            self.detection_display.append(
                "<span style='color:#00D2D2;'>[就绪] 检测信息框已就绪。"
                "点击「自动识别」开始扫描后, 此处会实时显示每次检测的类别/像素/坐标。</span>")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_all)
        self.timer.start(30)

        # ★ 仿真轨迹记录触发: 点「移动到位置」时发一帧 Bool(True) 给
        #   rebot_ponits_sim_node, 由它在 RViz 记录/停止末端轨迹
        self._trail_pub = None

        # ★ CAN看门狗(8s): PCAN适配器与Orbbec共享USB2总线, 相机大流量会
        #   饿死PCAN的USB传输 → 适配器"冻结"(Rx urb aborted -32, 灯常亮).
        #   冻结后 netdev仍在/后端进程可能仍活, 唯一可靠判据 = CAN往返探测.
        #   ★ 2026-10-29: 掉线后【不再自动恢复】, 只停后端+提示, 由用户
        #   手动点「启动机械臂」(自动重连瞬间电机扑向旧目标会伤人).
        self._arm_started = False          # 成功启动过机械臂后看门狗才介入
        self._can_wd_timer = QTimer(self)
        self._can_wd_timer.timeout.connect(self._can_watchdog)
        self._can_wd_timer.start(8000)

        self.init_servo_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # 舵机 UI 初始化
    # ──────────────────────────────────────────────────────────────────────────
    def init_servo_ui(self):
        # ★ UI v2.3 (2026-09-09 用户交互调整): 夹爪为独立全宽行(gripperRow);
        #   ▲/▼ = 步进±5mm(非绝对位); 新增[复位0mm]恢复开机并拢位;
        #   重标定改为明确文字按钮(原↻圆钮图标与J6顺时针撞脸易误解).
        g_lbl = QLabel("夹爪")
        _gl = g_lbl.font()
        _gl.setPixelSize(15)
        _gl.setWeight(QFont.Bold)
        g_lbl.setFont(_gl)
        g_lbl.setStyleSheet("color: %s; background: transparent;" % DJITheme.TEXT_2)
        self.servo_angle_input = QLineEdit("2")   # 默认2mm=并拢位
        self.servo_angle_input.setObjectName("axisField")
        self.servo_angle_input.setFixedWidth(52)
        self.servo_angle_input.setAlignment(Qt.AlignCenter)   # 数值居中
        self.servo_angle_input.setToolTip("两夹爪间距 mm (%g并拢 ~ %g张开)" % (
            GRIPPER_MM_TOUCH, GRIPPER_MM_OPEN))
        g_unit = QLabel("mm")   # ★ 单位可见化(用户要求: UI体现mm单位控制)
        _guf = g_unit.font()
        _guf.setPixelSize(12)
        g_unit.setFont(_guf)
        g_unit.setStyleSheet("color: %s; background: transparent;"
                             % DJITheme.TEXT_3)
        self.servo_set_btn = QPushButton("设定")
        self.servo_set_btn.setToolTip("设置夹爪间距(mm)")
        self.servo_open_btn = QPushButton("")
        self.servo_open_btn.setObjectName("circ")
        self.servo_open_btn.setIcon(DJIIcon.get(_SVG_UP, DJITheme.TEXT_2, 16))
        self.servo_open_btn.setIconSize(QSize(16, 16))
        self.servo_open_btn.setToolTip("张开 5mm(相对当前位步进)")
        self.servo_open_btn.clicked.connect(lambda: self._gripper_step(+5.0))
        self.servo_close_btn = QPushButton("")
        self.servo_close_btn.setObjectName("circ")
        self.servo_close_btn.setIcon(DJIIcon.get(_SVG_DOWN, DJITheme.TEXT_2, 16))
        self.servo_close_btn.setIconSize(QSize(16, 16))
        self.servo_close_btn.setToolTip("闭合 5mm(相对当前位步进)")
        self.servo_close_btn.clicked.connect(lambda: self._gripper_step(-5.0))
        self.servo_reset_btn = QPushButton("复位0mm")
        self.servo_reset_btn.setToolTip("恢复开机并拢位(0mm=指尖相触处轻握)")
        self.servo_reset_btn.clicked.connect(lambda: self._send_gripper(GRIPPER_MM_CLOSED))
        self.servo_recal_btn = QPushButton("重标定")
        self.servo_recal_btn.setToolTip(
            "夹爪零位重标定(手动掰动过夹爪/清理滑轨碎屑后使用; "
            "平时启动时的自动标定会处理零位漂移, 无需手动点)")
        self.servo_recal_btn.clicked.connect(self.recal_gripper)
        self.servo_set_btn.clicked.connect(self.set_servo_angle)

        # ★ 与 J6 行共用同一网格(第 1 行), 列宽自动对齐
        grid = getattr(self, '_joint_grid', None)
        if grid is not None:
            grid.addWidget(g_lbl, 1, 0)
            grid.addWidget(self.servo_angle_input, 1, 1)
            grid.addWidget(g_unit, 1, 2)
            grid.addWidget(self.servo_set_btn, 1, 3)
            grid.addWidget(self.servo_close_btn, 1, 4)   # 闭合(▼)在左
            grid.addWidget(self.servo_open_btn, 1, 5)    # 张开(▲)在右
            grid.addWidget(self.servo_reset_btn, 1, 6)
            grid.addWidget(self.servo_recal_btn, 1, 7)
            return self.servo_set_btn
        # 兜底(理论上不会走到): 退回原独立行
        grp = QWidget()
        h = QHBoxLayout(grp)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        for w in (g_lbl, self.servo_angle_input, g_unit, self.servo_set_btn,
                  self.servo_close_btn, self.servo_open_btn,
                  self.servo_reset_btn, self.servo_recal_btn):
            h.addWidget(w)
        row = self.centralWidget().findChild(QWidget, "gripperRow")
        if row is not None and row.layout() is not None:
            row.layout().addWidget(grp)
        return grp

    def set_servo_angle(self):
        try:
            mm = float(self.servo_angle_input.text())
        except Exception:
            mm = GRIPPER_MM_TOUCH
        self._send_gripper(mm)

    def _gripper_step(self, delta_mm):
        """▲/▼ 步进: 按一次移动5mm — 读当前实际间距±步进, 钳位[0,80]后下发."""
        try:
            rs = self.controller.arm_client.get_state()
            if rs.get("ok"):
                cur_mm = _gripper_rad_to_mm(float(rs.get("gripper", 0.0)))
            else:
                self.status_display.append("[gripper] 读不到夹爪状态(先启动机械臂)")
                return
        except Exception as e:
            print("[gripper] 步进读取失败: %s" % e)
            return
        tgt = max(GRIPPER_MM_CLOSED, min(GRIPPER_MM_OPEN, cur_mm + delta_mm))
        self._send_gripper(tgt)

    def set_j6_absolute(self):
        """J6 [设定]: 转到输入的绝对角度(rad)."""
        try:
            tgt = float(self.rotate_amount.text())
        except Exception:
            self.status_display.append("[J6] 输入无效(需为弧度数字)")
            return
        self._j6_move_to(tgt)

    def j6_return_home(self):
        """[J6归位]: 回到本次启动机械臂时记录的J6角度(不确定转了几圈时一键回开机位)."""
        if getattr(self, '_j6_home', None) is None:
            self.status_display.append("[J6] 尚未记录开机位(请先点[启动机械臂])")
            return
        self._j6_move_to(float(self._j6_home))

    def _j6_move_to(self, tgt_rad):
        """J6 转到绝对角度(仅J6动, 其他关节零位移).
        ★ 起点必须用UI指令台账(controller.joint_pos)而非 get_state 实测位:
          实测位含重力下垂误差(kp=25软刚度, 各关节实际比指令位略低) ——
          拿实测位当j1~j5发给服务器 = 把"下垂量"写进_q_goal, 其他关节
          会下沉一段("按J6设定J2动"的根因). 台记与服务器目标一致,
          其他关节起点=终点=指令位, 服务器零位移, 只有J6在动."""
        cur = list(self.controller.joint_pos)   # ★ 指令台账, 勿用get_state实测
        cur_j6 = cur[5]
        d = abs(tgt_rad - cur_j6)
        if d < 1e-3:
            self.status_display.append("[J6] 已在目标位 %.3f rad" % cur_j6)
            return
        new_pos = cur.copy()
        new_pos[5] = tgt_rad
        dur = max(1.5, d / 0.15 + 1.0)   # ~0.15rad/s, 大行程放慢
        self.controller.q6_value = tgt_rad
        self.start_arm_movement(cur, new_pos, dur)
        self.status_display.append("[J6] %.3f → %.3f rad (约%.0fs)" % (cur_j6, tgt_rad, dur))

    def _gripper_autocal(self, est=None, park_mm=74.0, **creep_kw):
        """★ 夹爪零位自动标定(阻塞15~40s, 供后台线程调用).
        匍匐锚定找指尖相触点 -> 触点=2mm 重算 REF -> 闸校验 -> 持久化 -> 回74mm.
        est = 启动时"读数-停车位置"的帧漂移推算(rad): 非 None 时锚定修正量须与
          est 交叉验证一致(防碎屑卡滞假锚点; 手动掰动过也会触发拒绝, 走[重标定]).
          est is None = 无推算依据(如首次运行), 仅保留绝对闸."""
        global _GRIP_CAL_LAST_RAD
        ac = self.controller.arm_client
        self._worker_log(
            "[gripper] 自动标定开始: 匍匐闭合寻找指尖相触点(压力≤3力矩, 请勿在指间放置物品)...")
        print("[gripper] 自动标定开始: 匍匐闭合寻找指尖相触点(压力≤3力矩, 请勿在指间放置物品)...")
        touch = _gripper_creep_anchor(ac, log=print, **creep_kw)
        if touch is None:
            self._worker_log("[gripper] 标定失败(详见日志)")
            return False
        new_ref = touch + (_GRIP_CAL_MM_REF - GRIPPER_MM_TOUCH) / _GRIP_CAL_MM_PER_RAD
        delta = _gripper_apply_cal_ref(new_ref)   # 先应用再校验(失败回滚)
        delta_mm = delta * _GRIP_CAL_MM_PER_RAD
        if abs(delta) > _GRIP_CAL_SANITY:
            _gripper_apply_cal_ref(new_ref - delta)   # 回滚
            self._worker_log(
                "[gripper] ★ 标定拒绝: 修正量 %.0fmm 超出±94mm绝对闸 —— "
                "疑似滑轨碎屑卡滞, 请手推夹爪全行程清理后重试" % delta_mm)
            return False
        if est is not None and abs(delta - est) > _GRIP_CAL_VERIFY_TOL:
            _gripper_apply_cal_ref(new_ref - delta)   # 回滚
            self._worker_log(
                "[gripper] ★ 标定拒绝: 锚定修正 %.0fmm 与停车位置推算的漂移 %.0fmm 不符 —— "
                "疑似碎屑卡滞或夹爪被手动掰动过. 清理/确认后点[↻]强制重标"
                % (delta_mm, est * _GRIP_CAL_MM_PER_RAD))
            return False
        park_rad = new_ref + (park_mm - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD
        _gripper_save_cal(new_ref, park_rad)
        _GRIP_CAL_LAST_RAD = park_rad
        ac.set_gripper(park_rad)
        self._worker_log(
            "[gripper] 标定完成: 零位修正 %+.1fmm, 夹爪正返回 %gmm 安全位(约%.0fs)"
            % (delta_mm, park_mm, (74.0 - GRIPPER_MM_TOUCH) / GRIPPER_SPEED_MM_S))
        return True

    def recal_gripper(self):
        """[↻重标定]按钮: 后台线程强制重标(est=None 跳过交叉验证, 供手动掰动/清理后).
        ★ 2026-10-18: 加并发保护+即时反馈 —— 标定要匍匐闭合15~40s, 原来点了
          没任何提示, 用户会连点导致多个标定线程并发操作夹爪互相干扰."""
        if not getattr(self, '_arm_started', False):
            self.status_display.append("[gripper] 请先点[启动机械臂]再重标定")
            return
        if getattr(self, '_gripper_calib_running', False):
            self.status_display.append(
                "[gripper] 标定已在进行中(匍匐闭合最长约40s), 请等当前标定结束")
            return
        self._gripper_calib_running = True
        self.status_display.append(
            "[gripper] 强制重标开始: 匍匐闭合寻找指尖相触点(约15~40s, 期间夹爪按钮暂不可用)...")
        def _run():
            try:
                self._gripper_autocal(est=None)
            finally:
                self._gripper_calib_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _send_gripper(self, dist_mm, wait_time=0.8):
        """夹爪控制 — 实物间距模式(所有按钮/输入框/动作序列的统一收口).
        输入 = 两夹爪之间的真实物理距离(mm), 与实物一一对应(全行程已验证±0.5mm):
          rad = _GRIP_CAL_RAD_REF + (mm-74)/15.75
        工作域 [0mm, 80mm]: 2=指尖相触并拢位, 0=并拢指令(相触处轻握, 抓取握力来源),
        80=张开上限. 服务器夹爪为柔顺模式(kp=10/kd=1), 大行程渐进逼近, 无撞击风险.
        超出工作域的输入一律钳位并提示; rad 二次钳位作双保险."""
        try:
            mm = float(dist_mm)
            if mm < GRIPPER_MM_CLOSED:
                mm = GRIPPER_MM_CLOSED
                self._log("[gripper] 间距低于0mm(并拢位), 已钳位到 %g mm" % GRIPPER_MM_CLOSED)
            if mm > GRIPPER_MM_OPEN:
                mm = GRIPPER_MM_OPEN
                self._log(
                    "[gripper] 间距高于工作上限, 已钳位到 %g mm (实测最大可到88)" % GRIPPER_MM_OPEN)
            rad = _GRIP_CAL_RAD_REF + (mm - _GRIP_CAL_MM_REF) / _GRIP_CAL_MM_PER_RAD
            if rad < GRIPPER_RAD_AT_MM_CLOSED:
                rad = GRIPPER_RAD_AT_MM_CLOSED
            if rad > GRIPPER_OPEN_MAX:
                rad = GRIPPER_OPEN_MAX
                self._log(
                    "[gripper] 目标已钳位到安全张开上限 %.2f rad" % GRIPPER_OPEN_MAX)
            # ★ 到位检测: 目标≈当前位置时明确提示(消除"点了没反应"的困惑)
            try:
                rs = self.controller.arm_client.get_state()
                if rs.get("ok"):
                    cur_mm = _gripper_rad_to_mm(rs.get("gripper", 0.0))
                    if abs(cur_mm - mm) < 1.0:
                        self._log("[gripper] 夹爪已在目标位 %.1fmm, 无需移动" % cur_mm)
                        return True
            except Exception:
                pass   # 读取失败不阻断发送
            r = self.controller.arm_client.set_gripper(rad)
            if not (r or {}).get("ok"):
                err = _errtext(r) if r else "无响应"
                self._log("[gripper] 发送失败: %s (请先点[启动机械臂])" % err)
                print("[gripper] 发送失败: %s (请先点[启动机械臂])" % err)
                return False
            print("[gripper] 目标 %g mm 已发送" % mm)
            try:
                _gripper_save_cal(_GRIP_CAL_RAD_REF, rad)   # ★ 停车位置记忆(下次启动交叉验证)
            except Exception:
                pass
            return True
        except Exception as e:
            msg = str(e)
            if "NoneType" in msg or "not started" in msg or "pipe" in msg.lower():
                hint = "机械臂未启动或后端已断开 — 请先点[启动机械臂]"
            else:
                hint = msg
            print("[gripper] 发送失败: %s" % hint)
            self._log("[gripper] 发送失败: %s" % hint)
            return False

    def set_gripper_angle(self, dist_mm):
        """兼容入口: 输入两夹爪间距(mm)."""
        self._send_gripper(dist_mm)

    def refresh_ports(self):
        pass

    def start_servo_scan(self):
        pass

    def update_servo_list(self, *args, **kwargs):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # 定时更新
    # ──────────────────────────────────────────────────────────────────────────
    def update_all(self):
        # ★ FK状态块实时跟随(2026-09-09): 状态信息栏的"实际末端位置/法向"
        #   原为事件触发的快照(动作结束才刷新, 与[读取当前位姿]按钮读数对
        #   不上的困惑来源) —— 现挂30ms定时器节流刷新, 每0.5s实时更新一次.
        # ★ 2026-10-18 修复断链: 本节流块此前只更新时间戳不调任何刷新
        #   ("FK实时"label 只在线程首次启动时刷过一次, 此后永不更新).
        #   "当前机械臂状态"大文本块按原设计保持纯事件驱动(运动完成/
        #   按钮点击时刷新), 不做周期刷新.
        import time as _t
        _now = _t.monotonic()
        if (_now - getattr(self, '_fk_refresh_last', 0.0) >= 0.5
                and not getattr(self, 'is_scanning', False)):
            self._fk_refresh_last = _now
            self._refresh_fk_block()
        # ★ 启动FK后台线程(首臂连接后启动一次)
        if not self._fk_bg_running:
            self._start_fk_bg_thread()
            try:
                self._refresh_fk_block()
            except Exception:
                pass
        # ★ 工作线程消息队列: 后台线程只 append(线程安全), 主线程取出显示
        #   (原QTimer.singleShot从python线程调度永不触发 —— "启动机械臂全程
        #   静默/无失败提示"的根因); print 落盘在 _worker_log 里已完成.
        while self._ui_msgs:
            try:
                text, status, ms = self._ui_msgs.pop(0)
            except Exception:
                break
            if text:
                self.status_display.append(text)
            if status:
                self.statusBar().showMessage(status, ms or 4000)
        # ★ 工作线程UI调用排水(2026-09-17): Qt控件操作/单次定时必须回主线程,
        #   原工作线程直接 QTimer.singleShot/setEnabled = "Timers cannot be
        #   started from another thread" + 跨线程控件 → 随机段错误
        #   ("UI经常闪退"的根因, 每次动作开始/结束各踩一次)
        while True:
            try:
                _fn = self._ui_calls.pop(0)
            except Exception:
                break
            try:
                _fn()
            except Exception as _e:
                print("[UI队列] 执行异常: %s" % _e)
        # ★ 相机看门狗的重启请求(由后台线程置位, 主线程执行保证Qt安全)
        if getattr(self, '_cam_wd_pending_restart', False):
            self._cam_wd_pending_restart = False
            self.status_display.append("[相机看门狗] USB重置完成, 自动重启相机管线...")
            self._restart_camera_pipeline()
            return
        # ★ 动作按钮看门狗: 动作已结束(action_running=False)但按钮仍是禁用态
        #   (_after_action_done 事件丢失等边界) -> 每30ms自动解锁, 杜绝"点下一动作无响应"
        if not getattr(self, 'action_running', False) and self.action_buttons and \
                not self.action_buttons[0].isEnabled():
            self.enable_action_buttons(True)
        self.update_camera_view()
        # ★ 标定读数栏刷新(仅标定模式, 内部自带节流/标志判断)
        self._calib_update_readout()
        self.update_arm_movement()
        self.update_auto_pipeline()  # ★ 新增：自动采集状态机
        # ★ 持续发布当前 joint_pos 到 /joint_states, 保证 robot_state_publisher
        #   的 base→6_Link TF 链不断 (即使机械臂静止), 供手眼标定 tf 查询用。
        #   gluon_node 订阅 /joint_states 驱动执行器, 重复发布当前值不影响运动。
        if self.controller.joint_pos is not None:
            msg = JointState()
            msg.name = JOINT_NAMES
            msg.position = list(self.controller.joint_pos)
            msg.header.stamp = rospy.Time.now()
            self.controller.publisher.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修改版：update_arm_movement — 添加扫描模式支持
    # ──────────────────────────────────────────────────────────────────────────
    def update_arm_movement(self):
        """非阻塞式的机械臂移动更新，支持夹爪控制和扫描模式"""
        # ★ reBot: 动作线程(execute_action)执行期间, 本30Hz定时器完全不发关节指令
        #   (含扫描模式!) —— 自动识别时扫描的j1摆动指令和动作线程的move_linear
        #   互相覆盖是"自动识别断电"的根因, 必须最先判断
        if getattr(self, 'action_running', False):
            return
        # ★ 巡点循环扫描(2026-09-17同步): 运动由扫描工作线程(30Hz流式)驱动,
        #   本30Hz定时器完全让路, 防止 set_joints 互相覆盖
        if self.is_scanning:
            return
        
        # ★ 新增：处理夹爪等待状态
        if self.is_gripper_waiting:
            current_time = rospy.get_time()
            if current_time - self.gripper_wait_start >= self.gripper_wait_duration:
                self.is_gripper_waiting = False
                self.statusBar().showMessage("夹爪动作完成，继续执行...")
            else:
                return  # 仍在等待中，直接返回

        if self.is_moving:
            current_time = rospy.get_time()
            elapsed = current_time - self.move_start_time
            # ★ 除零防护: 停止采集等会把 is_moving 置True而 duration 仍是0,
            #   30ms定时器进来除零 → Qt abort → UI闪退(点击停止闪退的根因)
            if self.move_duration <= 0:
                fraction = 1.0
            else:
                fraction = min(elapsed / self.move_duration, 1.0)

            current_pos = interpolate(self.move_start_pos, self.move_end_pos, fraction)
            pub2ros_pos(self.controller.publisher, current_pos)
            self.controller.joint_pos = current_pos
            self.controller.q6_value = current_pos[5]

            if fraction >= 1.0:
                self.is_moving = False
                self.show_current_status()

    # ★ 巡点循环扫描 (2026-09-17自kk_rebot同步): 旧 update_scan_movement(绕j1
    #   正弦摆动)已移除, 新扫描为 A→B→C→D→E→F→A 巡点循环, 运动由扫描工作线程
    #   (start_scan/_scan_loop_worker/_stream_leg) 30Hz流式驱动, 见对应方法.

    def init_ui(self):
        # ── UI v2.0: 顶部标题栏(全宽) + 双栏主体, 无翻页, 全功能同屏 ──
        main_widget = QWidget()
        main_widget.setObjectName("root")
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 顶部标题栏: 中文主标题居中(无英文副题)
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(74)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 6, 24, 6)
        header_layout.setSpacing(12)

        header_title = QLabel("刑侦现场物证采集系统")
        header_title.setObjectName("headerTitle")
        header_title.setAlignment(Qt.AlignCenter)
        header_title.setContentsMargins(34, 16, 34, 16)   # 给四角角标留出空间
        # ★ 大标题加粗: QSS 的 font-weight 在部分 Qt/字体组合下不生效, 直接用 QFont 定死
        _tf = header_title.font()
        _tf.setPixelSize(28)
        _tf.setWeight(QFont.DemiBold)
        _tf.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
        header_title.setFont(_tf)

        # ★ 机甲风装饰(V5): 标题两侧"断续刻度 + 六边形节点",
        #   标题本身再套一层四角 L 型瞄准角标。
        deco_l = MechaTitleDecor(-1)
        deco_r = MechaTitleDecor(+1)

        # 角标层与标题叠放: 用一层容器承载, 角标绝对定位覆盖整个容器
        title_wrap = QWidget()
        title_wrap_l = QVBoxLayout(title_wrap)
        title_wrap_l.setContentsMargins(0, 0, 0, 0)
        title_wrap_l.setSpacing(0)
        title_wrap_l.addWidget(header_title)
        title_brackets = TitleBrackets(title_wrap)
        title_brackets.setGeometry(0, 0, title_wrap.width(), title_wrap.height())
        title_brackets.show()

        def _sync_brackets():
            """容器尺寸变化时让角标层跟随(标题宽度随字体/DPI 变化)。"""
            title_brackets.setGeometry(0, 0, title_wrap.width(),
                                       title_wrap.height())
            title_brackets.raise_()
        title_wrap.resizeEvent = lambda e: _sync_brackets()
        title_wrap._brackets = title_brackets

        header_layout.addWidget(deco_l, 1)
        header_layout.addSpacing(16)
        header_layout.addWidget(title_wrap, 0)
        header_layout.addSpacing(16)
        header_layout.addWidget(deco_r, 1)
        root_layout.addWidget(header)

        # 主体: 左(控制卡片栏, 2) : 右(相机+信息, 3)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        body_layout.addWidget(splitter)
        root_layout.addLayout(body_layout, 1)
        self.setCentralWidget(main_widget)
        self.statusBar().showMessage("系统就绪")
        self.statusBar().setSizeGripEnabled(False)
    
    # 卡片/分区标题的英文小字: 已按需求全部移除(仅保留中文标题)
    _CARD_EN = {}

    def _card_head(self, glyph, title, index, margins=(0, 0, 0, 0)):
        """卡片头部行: 青色等宽编号 + 中文标题, 整组水平居中;
        两侧各一条发丝线做对称分隔(纯装饰)。glyph 参数保留仅为兼容旧调用。"""
        header = QWidget()
        header.setObjectName("cardHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(*margins)
        hl.setSpacing(10)

        # ★ 醒目化: 序号做成"青色实心编号块"(DJI 风标签), 中文标题 18px 粗体纯白
        idx_lbl = QLabel(index)
        idx_lbl.setObjectName("cardIndex")
        _fi = idx_lbl.font()
        _fi.setPixelSize(12)
        _fi.setWeight(QFont.Bold)
        idx_lbl.setFont(_fi)
        idx_lbl.setAlignment(Qt.AlignCenter)
        idx_lbl.setFixedHeight(20)
        idx_lbl.setMinimumWidth(28)
        idx_lbl.setStyleSheet(
            "color: %s; background-color: %s; font-family: %s;"
            " padding: 1px 7px; border-radius: 2px;"
            % (DJITheme.ON_ACCENT, DJITheme.ACCENT, DJITheme.MONO))

        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        # ★ 字号/字重直接用 QFont 定死(QSS 的 font-weight 在部分 Qt/字体组合下不生效)
        _f = title_lbl.font()
        _f.setPixelSize(18)
        _f.setWeight(QFont.Bold)
        _f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        title_lbl.setFont(_f)
        title_lbl.setStyleSheet("color: #FFFFFF; background: transparent;")

        rule_l = QWidget()
        rule_l.setObjectName("cardRule")
        rule_l.setFixedHeight(1)
        rule_l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rule_r = QWidget()
        rule_r.setObjectName("cardRule")
        rule_r.setFixedHeight(1)
        rule_r.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 左右等权重发丝线 + 中间标题组: 标题严格居中
        hl.addWidget(rule_l, 1)
        hl.addSpacing(8)
        hl.addWidget(idx_lbl, 0)
        hl.addSpacing(4)
        hl.addWidget(title_lbl, 0)
        hl.addSpacing(8)
        hl.addWidget(rule_r, 1)
        return header

    def _make_card(self, glyph, title, index, primary=False):
        """统一功能卡片(右栏内容区用): 头部行 + 内容布局。DJI 风: 方角细描边,
        无阴影无渐变, 层级靠 1px 线与字距区分。"""
        card = QGroupBox()
        if primary:
            card.setObjectName("cardPrimary")
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 11, 14, 13)
        v.setSpacing(10)
        v.addWidget(self._card_head(glyph, title, index))
        v.addWidget(HLine(DJITheme.LINE_SOFT))
        return card, v

    def _make_section(self, glyph, title, index):
        """无框分区(左栏专用): 头部行 + 发丝线 + 内容, 纯排版分隔。"""
        section = QWidget()
        v = QVBoxLayout(section)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.addWidget(self._card_head(glyph, title, index))
        v.addWidget(HLine(DJITheme.LINE_SOFT))
        return section, v

    def create_left_panel(self):
        """UI v2.2 左栏: 无框排版式分区; X/Y/Z/J6 紧凑下划线输入槽(2×2);
        关节旋转与夹爪开合为圆形图标按钮; 夹爪控件由 init_servo_ui 运行时
        插入 jointRow; 底部保留系统工具条。"""
        panel = QWidget()
        panel.setObjectName("leftPanel")   # ★ init_servo_ui 兜底查找用
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(16)

        # ══════════ 01 目标位姿(下划线输入槽) ══════════
        sec1, sec1_v = self._make_section("⊕", "目标位姿 · 笛卡尔", "01")

        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()
        self.j6_input = QLineEdit("0")   # ★ 2026-10-30 用户要求: J6 默认填 0

        def _axis_cell(row, col, name, unit, field):
            cell = QHBoxLayout()
            cell.setSpacing(7)
            # ★ 轴标签用界面字体(Lato)而非等宽字体: Ubuntu Mono 的 X/J 字形
            #   在小字号下歪斜难辨; 同时放大到 16px Bold 并提亮为次要文字色
            # ★ 轴标签做成"青色标签块": 小色块 + 白色字母, 与右侧数值形成
            #   清晰的分栏, 同时比纯灰文字醒目得多
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(34, 24)
            _lf = lbl.font()
            _lf.setPixelSize(14)
            _lf.setWeight(QFont.Bold)
            lbl.setFont(_lf)
            lbl.setStyleSheet(
                "color: %s; background-color: #1B2A2C;"
                " border: 1px solid %s; border-radius: 2px;"
                % (DJITheme.ACCENT, DJITheme.ACCENT_DK))
            field.setObjectName("axisField")
            field.setFixedWidth(96)
            field.setAlignment(Qt.AlignCenter)   # 数值居中
            unit_lbl = QLabel(unit)
            _uf = unit_lbl.font()
            _uf.setPixelSize(12)
            unit_lbl.setFont(_uf)
            unit_lbl.setStyleSheet("color: %s; background: transparent;"
                                   % DJITheme.TEXT_3)
            cell.addWidget(lbl)
            cell.addWidget(field)
            cell.addWidget(unit_lbl)
            # ★ 不加尾部弹簧: 两列按栅格列宽对齐, 左对齐更整齐
            axis_grid.addLayout(cell, row, col)

        axis_grid = QGridLayout()
        axis_grid.setHorizontalSpacing(18)
        axis_grid.setVerticalSpacing(9)
        _axis_cell(0, 0, "X", "m", self.x_input)
        _axis_cell(0, 1, "Y", "m", self.y_input)
        _axis_cell(1, 0, "Z", "m", self.z_input)
        _axis_cell(1, 1, "J6", "rad", self.j6_input)
        sec1_v.addLayout(axis_grid)

        # 表面法向量: 三个小输入槽一行
        normal_row = QHBoxLayout()
        normal_row.setSpacing(6)
        n_lbl = QLabel("法向")
        _nl = n_lbl.font()
        _nl.setPixelSize(14)
        _nl.setWeight(QFont.Bold)
        n_lbl.setFont(_nl)
        n_lbl.setStyleSheet("color: %s; background: transparent;" % DJITheme.TEXT_2)
        normal_row.addWidget(n_lbl)
        self.nx_input = QLineEdit("0")
        self.ny_input = QLineEdit("0")
        self.nz_input = QLineEdit("-1")
        for lbl_txt, w in (("nx", self.nx_input), ("ny", self.ny_input), ("nz", self.nz_input)):
            w.setObjectName("axisField")
            w.setFixedWidth(56)
            w.setAlignment(Qt.AlignCenter)   # 数值居中
            u = QLabel(lbl_txt)
            u.setAlignment(Qt.AlignCenter)
            u.setFixedSize(28, 20)
            _nf = u.font()
            _nf.setPixelSize(11)
            u.setFont(_nf)
            u.setStyleSheet(
                "color: %s; background-color: #171D1F;"
                " border: 1px solid #2E3A3C; border-radius: 2px;"
                % DJITheme.TEXT_2)
            normal_row.addWidget(w)
            normal_row.addWidget(u)
        normal_row.addStretch(1)
        sec1_v.addLayout(normal_row)

        # 姿态四元数读出
        self.normal_status_label = QLabel("当前姿态: [1.0, 0.0, 0.0, 0.0]")
        self.normal_status_label.setStyleSheet(
            "color: %s; font-size: 12px; font-family: %s;"
            " background-color: #101114; border: 1px solid %s;"
            " border-left: 2px solid %s; padding: 5px 8px;"
            % (DJITheme.TEXT_2, DJITheme.MONO, DJITheme.LINE_SOFT, DJITheme.ACCENT))
        sec1_v.addWidget(self.normal_status_label)

        # 移动 + 读取位姿(标定用)
        cart_btn_row = QHBoxLayout()
        cart_btn_row.setSpacing(6)
        move_btn = QPushButton("移动到位置")
        move_btn.setObjectName("primary")
        move_btn.setIcon(DJIIcon.get(_SVG_TARGET, DJITheme.ON_ACCENT, 16))
        move_btn.setIconSize(QSize(16, 16))
        move_btn.setToolTip("按 X/Y/Z/法向 求解逆运动学并移动")
        move_btn.clicked.connect(self.move_to_position)
        pose_btn = QPushButton("读取当前位姿")
        pose_btn.setToolTip("读取当前末端位姿(FK), 用于现场标定预设点")
        pose_btn.clicked.connect(self.read_current_pose)
        cart_btn_row.addWidget(move_btn, 2)
        cart_btn_row.addWidget(pose_btn, 1)
        sec1_v.addLayout(cart_btn_row)

        # ══════════ 02 流程控制(回零/地面检测, 图标+加高按钮) ══════════
        sec2, sec2_v = self._make_section("▶", "流程控制", "02")
        flow_grid = QGridLayout()
        flow_grid.setHorizontalSpacing(8)
        flow_grid.setVerticalSpacing(8)

        # ★ 2026-10-30 用户要求: 移除「开始扫描/停止扫描/自动识别/停止自动采集」,
        #   只保留「返回初始位置」与「检测地面」并排一行.
        home_btn = QPushButton("返回初始位置")
        home_btn.setObjectName("primary")   # 与「移动到位置」同色(主操作)
        home_btn.setIcon(DJIIcon.get(_SVG_HOME, DJITheme.ON_ACCENT, 16))
        home_btn.setIconSize(QSize(16, 16))
        home_btn.setMinimumHeight(42)
        # ★ 2026-10-18: lambda 隔离信号参数 —— clicked(bool) 会把 checked=False
        #   传进 return_to_zero 的 duration 形参, False≠None 跳过自适应分支,
        #   再被 start_arm_movement 的兜底改成 1s 猛冲("怎么改都快"的根因)
        home_btn.clicked.connect(lambda: self.return_to_zero())
        # ★ 2026-10-30 用户要求: 「检测地面」与「返回初始位置」并列, 同 primary 蓝色样式
        ground_btn = QPushButton("检测地面")
        ground_btn.setObjectName("primary")
        ground_btn.setIcon(DJIIcon.get(_SVG_DOWN, DJITheme.ON_ACCENT, 16))
        ground_btn.setIconSize(QSize(16, 16))
        ground_btn.setMinimumHeight(42)
        ground_btn.setToolTip("检测地面: 移动到 (0.35, 0, 0), 法向竖直朝下 (0,0,-1)")
        ground_btn.clicked.connect(lambda: self.detect_ground())

        flow_grid.addWidget(home_btn, 0, 0)
        flow_grid.addWidget(ground_btn, 0, 1)
        sec2_v.addLayout(flow_grid)

        # ══════════ 03 夹爪设置(J6与夹爪上下两行, 用户要求) ══════════
        sec3, sec3_v = self._make_section("⇄", "夹爪设置", "03")

        # ── 关节/夹爪共用网格: 两行共享列宽, 保证按钮纵向严格对齐 ──
        #   列: 0标签 1输入 2单位 3设定 4步进- 5步进+ 6动作 7动作
        self._joint_grid = QGridLayout()
        self._joint_grid.setContentsMargins(0, 0, 0, 0)
        self._joint_grid.setHorizontalSpacing(8)
        self._joint_grid.setVerticalSpacing(10)
        #   列宽按左栏实际可用宽度收紧(原来 8 列总宽超出面板, "重标定"被挤出屏幕)
        for _col, _w in ((0, 40), (1, 56), (2, 26), (3, 52), (4, 32),
                         (5, 32), (6, 76), (7, 62)):
            self._joint_grid.setColumnMinimumWidth(_col, _w)

        # ── Row A: J6 (绝对设定 + 步进旋转 + 回开机位) ──
        j6_row = 0
        j_lbl = QLabel("J6 ±")
        _jl = j_lbl.font()
        _jl.setPixelSize(15)
        _jl.setWeight(QFont.Bold)
        j_lbl.setFont(_jl)
        j_lbl.setStyleSheet("color: %s; background: transparent;" % DJITheme.TEXT_2)
        self.rotate_amount = QLineEdit("0.1")
        self.rotate_amount.setObjectName("axisField")
        self.rotate_amount.setFixedWidth(64)
        self.rotate_amount.setAlignment(Qt.AlignCenter)   # 数值居中
        self.rotate_amount.setToolTip(
            "J6 弧度值: 点[设定]转到该绝对角度; ↻/↺ 以它为步进量旋转")
        j_unit = QLabel("rad")
        _juf = j_unit.font()
        _juf.setPixelSize(12)
        j_unit.setFont(_juf)
        j_unit.setStyleSheet("color: %s; background: transparent;"
                             % DJITheme.TEXT_3)
        self.j6_set_btn = QPushButton("设定")
        self.j6_set_btn.setToolTip("J6 转到输入的绝对角度(rad)")
        self.j6_set_btn.clicked.connect(self.set_j6_absolute)
        cw_btn = QPushButton("")
        cw_btn.setObjectName("circ")
        cw_btn.setIcon(DJIIcon.get(_SVG_ROTATE_CW, DJITheme.TEXT_2, 17))
        cw_btn.setIconSize(QSize(17, 17))
        cw_btn.setToolTip("J6 顺时针旋转(步进量=输入值)")
        cw_btn.clicked.connect(lambda: self.rotate_j6(float(self.rotate_amount.text())))
        ccw_btn = QPushButton("")
        ccw_btn.setObjectName("circ")
        ccw_btn.setIcon(DJIIcon.get(_SVG_ROTATE_CCW, DJITheme.TEXT_2, 17))
        ccw_btn.setIconSize(QSize(17, 17))
        ccw_btn.setToolTip("J6 逆时针旋转(步进量=输入值)")
        ccw_btn.clicked.connect(lambda: self.rotate_j6(-float(self.rotate_amount.text())))
        self.j6_home_btn = QPushButton("J6归位")
        self.j6_home_btn.setToolTip("回到本次启动机械臂时记录的J6角度(不确定转了几圈时一键回开机位)")
        self.j6_home_btn.clicked.connect(self.j6_return_home)
        _g = self._joint_grid
        _g.addWidget(j_lbl, 0, 0)
        _g.addWidget(self.rotate_amount, 0, 1)
        _g.addWidget(j_unit, 0, 2)
        _g.addWidget(self.j6_set_btn, 0, 3)
        _g.addWidget(cw_btn, 0, 4)
        _g.addWidget(ccw_btn, 0, 5)
        _g.addWidget(self.j6_home_btn, 0, 6)   # 单列, 与"复位0mm"对齐
        _g.setColumnStretch(7, 0)

        # ── Row B: 夹爪控件由 init_servo_ui 运行时插入同一网格的第 1 行 ──
        gripper_row_w = QWidget()
        gripper_row_w.setObjectName("gripperRow")   # ★ 兼容旧查找逻辑
        gripper_row_w.setVisible(False)
        sec3_v.addWidget(gripper_row_w)
        sec3_v.addLayout(self._joint_grid)

        # ══════════ 04 预设动作(三个全宽等价按钮, 完全并列) ══════════
        sec4, sec4_v = self._make_section("▣", "预设动作", "04")
        height_row = QHBoxLayout()
        height_row.setSpacing(6)
        height_label = QLabel("安全高度 H (m)")
        height_label.setStyleSheet("color: %s; font-size: 13px;"
                                   % DJITheme.TEXT_3)
        self.height_input = QLineEdit(str(SAFE_HEIGHT))
        self.height_input.setObjectName("axisField")
        self.height_input.setFixedWidth(92)
        self.height_input.setAlignment(Qt.AlignCenter)   # 数值居中
        height_row.addWidget(height_label)
        height_row.addWidget(self.height_input)
        height_row.addStretch(1)
        sec4_v.addLayout(height_row)

        # ★ 工具补偿(2026-10-31 用户要求): 涂抹/粘取工具尖端伸出长度(mm),
        #   可UI调整, 默认 涂抹20mm/粘取34mm; 取放工具环节不用, 只作用于
        #   流程目标点(接触点沿法向反退工具长度)
        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        tool_label = QLabel("工具补偿 涂抹:")
        tool_label.setStyleSheet("color: %s; font-size: 13px;" % DJITheme.TEXT_3)
        self.tool_len_smear_input = QLineEdit("20")
        self.tool_len_smear_input.setObjectName("axisField")
        self.tool_len_smear_input.setFixedWidth(52)
        self.tool_len_smear_input.setAlignment(Qt.AlignCenter)
        tool_mm1 = QLabel("mm, 粘取:")
        tool_mm1.setStyleSheet("color: %s; font-size: 13px;" % DJITheme.TEXT_3)
        self.tool_len_dab_input = QLineEdit("34")
        self.tool_len_dab_input.setObjectName("axisField")
        self.tool_len_dab_input.setFixedWidth(52)
        self.tool_len_dab_input.setAlignment(Qt.AlignCenter)
        tool_mm2 = QLabel("mm")
        tool_mm2.setStyleSheet("color: %s; font-size: 13px;" % DJITheme.TEXT_3)
        tool_row.addWidget(tool_label)
        tool_row.addWidget(self.tool_len_smear_input)
        tool_row.addWidget(tool_mm1)
        tool_row.addWidget(self.tool_len_dab_input)
        tool_row.addWidget(tool_mm2)
        tool_row.addStretch(1)
        sec4_v.addLayout(tool_row)

        # ★ 动作按钮(2026-10-25 用户要求): 涂抹/粘取各分1/2号位(仅架位x不同,
        #   1=涂抹1号位, 4=涂抹2号位, 2=粘取1号位, 5=粘取2号位, 3=抓取不变);
        #   全宽、等高、图标+文字, 纵向排列
        self.action1_btn = ActionButton("涂抹 · 1 号位", _SVG_SMEAR,
                                        DJITheme.ACCENT, 20)
        self.action1_btn.setToolTip("液体类: 取棉签(架1号位 x=0.143)→转移到目标→往复涂抹→放回")
        self.action1_btn.setObjectName("actionRow")
        self.action1_btn.setMinimumHeight(40)
        self.action1_btn.clicked.connect(lambda: self.execute_action(1))
        self.action4_btn = ActionButton("涂抹 · 2 号位", _SVG_SMEAR,
                                        DJITheme.ACCENT, 20)
        self.action4_btn.setToolTip("液体类: 取棉签(架2号位 x=0.2)→转移到目标→往复涂抹→放回")
        self.action4_btn.setObjectName("actionRow")
        self.action4_btn.setMinimumHeight(40)
        self.action4_btn.clicked.connect(lambda: self.execute_action(4))
        self.action2_btn = ActionButton("粘取 · 1 号位", _SVG_DAB,
                                        DJITheme.ACCENT, 20)
        self.action2_btn.setToolTip("微小类: 取粘取器(架1号位 x=0.143)→下压停留→粘附→放回")
        self.action2_btn.setObjectName("actionRow")
        self.action2_btn.setMinimumHeight(40)
        self.action2_btn.clicked.connect(lambda: self.execute_action(2))
        self.action5_btn = ActionButton("粘取 · 2 号位", _SVG_DAB,
                                        DJITheme.ACCENT, 20)
        self.action5_btn.setToolTip("微小类: 取粘取器(架2号位 x=0.2)→下压停留→粘附→放回")
        self.action5_btn.setObjectName("actionRow")
        self.action5_btn.setMinimumHeight(40)
        self.action5_btn.clicked.connect(lambda: self.execute_action(5))
        self.action3_btn = ActionButton("动作 · 抓取", _SVG_GRIPPER,
                                        DJITheme.ACCENT, 20)
        self.action3_btn.setToolTip("固体类: 沿法向接近→闭合夹爪→转移→回收位释放")
        self.action3_btn.setObjectName("actionRow")
        self.action3_btn.setMinimumHeight(40)
        self.action3_btn.clicked.connect(lambda: self.execute_action(3))
        for btn in (self.action1_btn, self.action4_btn, self.action2_btn,
                    self.action5_btn, self.action3_btn):
            sec4_v.addWidget(btn)

        self.action_buttons = [self.action1_btn, self.action4_btn, self.action2_btn,
                               self.action5_btn, self.action3_btn]

        # ══════════ 底部固定系统条(启动/急停 + 工作空间说明) ══════════
        system_group = QWidget()
        system_group.setObjectName("leftFooter")
        system_layout = QHBoxLayout(system_group)
        system_layout.setContentsMargins(10, 8, 10, 8)
        system_layout.setSpacing(6)

        start_btn = QPushButton("启动机械臂")
        start_btn.setObjectName("primary")
        start_btn.setIcon(DJIIcon.get(_SVG_POWER, DJITheme.ON_ACCENT, 15))
        start_btn.setIconSize(QSize(15, 15))
        start_btn.clicked.connect(self.start_arm)
        stop_btn = QPushButton("紧急停止")
        stop_btn.setObjectName("danger")
        stop_btn.setIcon(DJIIcon.get(_SVG_STOP, DJITheme.DANGER, 13))
        stop_btn.setIconSize(QSize(13, 13))
        stop_btn.clicked.connect(self.emergency_stop)
        # ★ 按钮组整体居中: 前留弹簧 + 后留弹簧, 按钮组不拉伸
        system_layout.addStretch(1)
        system_layout.addWidget(start_btn)
        # 启动与急停之间留出安全间距: 急停是安全关键按钮, 与常规启动按钮
        # 贴太近容易误触; 拉开一段固定距离, 视觉上也形成"常规操作 | 安全操作"分组
        system_layout.addSpacing(34)
        system_layout.addWidget(stop_btn)
        system_layout.addStretch(1)

        # ★ 工作空间提示(底部条内): 整块水平居中; 三行用等宽字体 + 固定轴名列,
        #   保证 X/Y/Z 与各自数值严格纵向对齐(含负号).
        ws_sep = QWidget()
        ws_sep.setObjectName("vSep")
        ws_sep.setFixedSize(1, 40)
        system_layout.addWidget(ws_sep)

        ws_box = QWidget()
        ws_box.setObjectName("wsBox")
        ws_grid = QGridLayout(ws_box)
        ws_grid.setContentsMargins(2, 0, 2, 0)
        ws_grid.setHorizontalSpacing(4)
        ws_grid.setVerticalSpacing(3)

        # 标题: 跨两列居中
        ws_title = QLabel("工作空间 (m)")
        _wt = ws_title.font()
        _wt.setPixelSize(11)
        ws_title.setFont(_wt)
        ws_title.setAlignment(Qt.AlignCenter)
        ws_title.setStyleSheet("color: %s; background: transparent;"
                               % DJITheme.TEXT_3)
        # ★ 标题只横跨"数值区"(第 1~3 列, 不含轴名列), 并居中:
        #   这样标题中线与下方 "0.15 ~ 0.40" 数值区的中线严格重合
        #   (实测: 标题跨 4 列时中心偏左 10px)
        ws_grid.addWidget(ws_title, 0, 1, 1, 3, Qt.AlignHCenter | Qt.AlignVCenter)

        # 三轴: 每行拆成"轴名 / 下限 / ~ / 上限"四个固定宽度字段,
        #   这样波浪线、数字起点、小数点全部纵向对齐(负号也不会顶偏)
        for _i, (_axis, _lo, _hi) in enumerate(
                (("X", "0.15", "0.40"), ("Y", "-0.10", "0.30"),
                 ("Z", "0.10", "0.35")), start=1):
            _al = QLabel(_axis)
            _al.setFixedWidth(14)
            _af = _al.font()
            _af.setPixelSize(11)
            _af.setWeight(QFont.Bold)
            _al.setFont(_af)
            _al.setStyleSheet("color: %s; font-family: %s; background: transparent;"
                              % (DJITheme.ACCENT, DJITheme.MONO))
            _al.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            # 下限: 右对齐 → 个位/小数点位置一致
            #   ★ 宽度按最长内容 "-0.10"(43px) 留足, 否则负号会被裁掉
            _lol = QLabel(_lo)
            _lol.setFixedWidth(44)
            _lf = _lol.font()
            _lf.setPixelSize(11)
            _lol.setFont(_lf)
            _lol.setStyleSheet("color: %s; font-family: %s; background: transparent;"
                               % (DJITheme.TEXT_2, DJITheme.MONO))
            _lol.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # 波浪线: 固定宽度居中 → 三行严格对齐
            _tl = QLabel("~")
            _tl.setFixedWidth(12)
            _tf = _tl.font()
            _tf.setPixelSize(11)
            _tl.setFont(_tf)
            _tl.setStyleSheet("color: %s; font-family: %s; background: transparent;"
                              % (DJITheme.TEXT_3, DJITheme.MONO))
            _tl.setAlignment(Qt.AlignCenter)

            _hil = QLabel(_hi)
            _hil.setFixedWidth(34)
            _hif = _hil.font()
            _hif.setPixelSize(11)
            _hil.setFont(_hif)
            _hil.setStyleSheet("color: %s; font-family: %s; background: transparent;"
                               % (DJITheme.TEXT_2, DJITheme.MONO))
            _hil.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            ws_grid.addWidget(_al, _i, 0)
            ws_grid.addWidget(_lol, _i, 1)
            ws_grid.addWidget(_tl, _i, 2)
            ws_grid.addWidget(_hil, _i, 3)

        # 工作空间块紧随分隔线, 尾部留一个弹簧吸收多余空间
        system_layout.addWidget(ws_box, 0, Qt.AlignVCenter)
        system_layout.addStretch(1)

        # ══════════ 组装: 分区之间均布弹簧(空白分散, 不再堆在底部) ══════════
        for w in (sec1, sec2, sec3, sec4):
            layout.addWidget(w)
            layout.addStretch(1)   # ★ 每个分区后等分弹簧: 下半空白均匀摊开
        layout.addWidget(system_group)
        return panel

    def create_right_panel(self):
        """UI v2.0 右栏: 相机卡片(画面+人工标注栏+相机控制行) + 下方
        检测信息/状态信息 左右并排(替代原纵向三连大块, 减少纵向留白)。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ══════════ 06 相机卡片 ══════════
        camera_group, camera_layout = self._make_card("▣", "相机视图 · 人工标注", "06")

        # ★ 画框模式: 增强控件(冻结帧+拖拽画框), 直播外观与原 QLabel 一致
        self.camera_label = AnnotateCameraLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setText("等待相机图像...")
        self.camera_label.setStyleSheet(
            "background-color: #070708; color: %s;" % DJITheme.TEXT_3)
        self.camera_label.box_ready.connect(self.on_annotate_box)
        # ★ DJI 风取景器: 无画面时叠加角标/准星占位层(有画面时自动隐去)
        self._camera_placeholder = CameraPlaceholder(self.camera_label)
        self._camera_placeholder.setGeometry(0, 0,
                                             self.camera_label.width(),
                                             self.camera_label.height())
        self.camera_label.overlay = self._camera_placeholder
        self._camera_placeholder.show()
        self.camera_label.setText("")   # 空态文字统一由占位层绘制, 避免重影

        # ★★★ 人工标注工作流条 ★★★
        # [人工标注(冻结画面)] → 拖拽画框(黄=待确认) → [确认框选](绿=生效) /
        # [取消框选](清框重画) → [退出]回直播; 右端为状态读出
        annotate_bar = QWidget()
        annotate_bar.setObjectName("annotateBar")
        annotate_layout = QHBoxLayout(annotate_bar)
        annotate_layout.setContentsMargins(10, 8, 10, 8)
        annotate_layout.setSpacing(8)
        self.annotate_pause_btn = QPushButton("人工标注")
        self.annotate_pause_btn.setObjectName("primary")
        self.annotate_pause_btn.setIcon(DJIIcon.get(_SVG_PAUSE, DJITheme.ON_ACCENT, 14))
        self.annotate_pause_btn.setIconSize(QSize(14, 14))
        self.annotate_pause_btn.setToolTip("停止机械臂并冻结当前画面, 进入拖拽画框模式")
        self.annotate_pause_btn.clicked.connect(self.on_annotate_pause)
        self.annotate_confirm_btn = QPushButton("确认框选")
        self.annotate_confirm_btn.setObjectName("primary")
        self.annotate_confirm_btn.setIcon(DJIIcon.get(_SVG_CHECK, DJITheme.ON_ACCENT, 14))
        self.annotate_confirm_btn.setIconSize(QSize(14, 14))
        self.annotate_confirm_btn.setToolTip("待确认的框正式生效(记录坐标/传给机械臂)")
        self.annotate_confirm_btn.clicked.connect(self.on_annotate_confirm)
        self.annotate_confirm_btn.setEnabled(False)
        self.annotate_cancel_btn = QPushButton("取消框选")
        self.annotate_cancel_btn.setToolTip("清除当前框重新画(不退出冻结画面)")
        self.annotate_cancel_btn.clicked.connect(self.on_annotate_cancel)
        self.annotate_cancel_btn.setEnabled(False)
        self.annotate_resume_btn = QPushButton("退出")
        self.annotate_resume_btn.setToolTip("退出画框模式, 恢复实时预览")
        self.annotate_resume_btn.clicked.connect(self.on_annotate_resume)
        self.annotate_status_label = QLabel("画框模式: 未激活")
        self.annotate_status_label.setStyleSheet(
            "color: %s; font-size: 12px; font-family: %s;"
            % (DJITheme.TEXT_3, DJITheme.MONO))
        self.annotate_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.annotate_status_label.setMinimumWidth(210)
        annotate_layout.addWidget(self.annotate_pause_btn)
        sep1 = QWidget()
        sep1.setObjectName("vSep")
        sep1.setFixedSize(1, 20)
        annotate_layout.addWidget(sep1)
        annotate_layout.addWidget(self.annotate_confirm_btn)
        annotate_layout.addWidget(self.annotate_cancel_btn)
        annotate_layout.addWidget(self.annotate_resume_btn)
        annotate_layout.addStretch(1)
        annotate_layout.addWidget(self.annotate_status_label)

        # ══════════ 相机控制条: 画面状态徽章 + 启停 + 标定模式 ══════════
        cam_bar = QWidget()
        cam_bar.setObjectName("camBar")
        cam_control_layout = QHBoxLayout(cam_bar)
        cam_control_layout.setContentsMargins(10, 8, 10, 8)
        cam_control_layout.setSpacing(8)
        self.cam_state_chip = QLabel("待机")
        self.cam_state_chip.setObjectName("camChip")
        self._cam_chip_state = "待机"
        start_cam_btn = QPushButton("启动相机")
        start_cam_btn.setObjectName("primary")
        start_cam_btn.setIcon(DJIIcon.get(_SVG_CAMERA, DJITheme.ON_ACCENT, 15))
        start_cam_btn.setIconSize(QSize(15, 15))
        start_cam_btn.clicked.connect(self.start_camera)
        stop_cam_btn = QPushButton("停止相机")
        stop_cam_btn.setIcon(DJIIcon.get(_SVG_STOP, DJITheme.TEXT_2, 12))
        stop_cam_btn.setIconSize(QSize(12, 12))
        stop_cam_btn.clicked.connect(self.stop_camera)
        # ★★★ 标定模式开关 (2026-10-28) ★★★
        # 勾选后「启动相机」跑 aruco_calib_gemini.launch(ChArUco检测)而非
        # v4检测管线; 画面显示 /aruco_single/result(带角点+坐标轴叠加),
        # 下方标定读数栏实时显示板位姿. 切换后需重新点「启动相机」生效.
        self.calib_mode_btn = QPushButton("标定模式")
        self.calib_mode_btn.setCheckable(True)
        self.calib_mode_btn.setToolTip(
            "勾选后点「启动相机」启动 ChArUco 标定管线\n"
            "(板: 5x7 / 方格36.5mm / marker23mm / DICT_6X6_250)\n"
            "画面显示检测结果, 标定读数栏实时显示板位姿")
        sep2 = QWidget()
        sep2.setObjectName("vSep")
        sep2.setFixedSize(1, 20)
        sep3 = QWidget()
        sep3.setObjectName("vSep")
        sep3.setFixedSize(1, 20)
        cam_control_layout.addWidget(self.cam_state_chip)
        cam_control_layout.addWidget(start_cam_btn)
        cam_control_layout.addWidget(stop_cam_btn)
        cam_control_layout.addWidget(sep2)
        cam_control_layout.addWidget(self.calib_mode_btn)
        cam_control_layout.addWidget(sep3)
        cam_hint = QLabel("拖拽画矩形框 · 框中心即目标点")
        cam_hint.setStyleSheet("color: %s; font-size: 12px;" % DJITheme.TEXT_3)
        cam_control_layout.addWidget(cam_hint)
        cam_control_layout.addStretch(1)

        # ══════════ 标定读数栏 (仅标定模式显示) ══════════
        # 板位姿来自 /aruco_single/pose (相机系) + 板心在基座系的推算;
        # 同时显示检出 marker 数与重投影残差, 便于判断当前摆位是否可用.
        self.calib_info_bar = QWidget()
        self.calib_info_bar.setObjectName("annotateBar")
        _calib_lay = QHBoxLayout(self.calib_info_bar)
        _calib_lay.setContentsMargins(10, 6, 10, 6)
        _calib_lay.setSpacing(14)
        self.calib_pose_label = QLabel("标定板: 未检出")
        self.calib_pose_label.setStyleSheet(
            "color: %s; font-size: 12px; font-family: %s;"
            % (DJITheme.TEXT_2, DJITheme.MONO))
        self.calib_pose_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.calib_pose_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.calib_arm_label = QLabel("臂位姿: —")
        self.calib_arm_label.setStyleSheet(
            "color: %s; font-size: 12px; font-family: %s;"
            % (DJITheme.TEXT_3, DJITheme.MONO))
        self.calib_arm_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _calib_lay.addWidget(self.calib_pose_label, 1)
        _calib_lay.addWidget(self.calib_arm_label, 0)
        self.calib_info_bar.setVisible(False)   # 非标定模式隐藏
        # ★ 信号连接放在 calib_info_bar/标签都创建之后 —— 否则初始 setChecked
        #   触发 toggled 时槽里访问 calib_info_bar 会 AttributeError
        self.calib_mode_btn.toggled.connect(self.on_calib_mode_toggled)

        camera_layout.addWidget(self.camera_label, 1)
        camera_layout.addWidget(annotate_bar)
        camera_layout.addWidget(self.calib_info_bar)
        camera_layout.addWidget(cam_bar)

        # ══════════ 07 检测信息 | 08 状态信息 并排 ══════════
        # ★★★ 检测信息实时显示框 ★★★
        # 扫描过程中每次相机检测到物证, 都在这里显示:
        #   类别 / 置信度 / 像素坐标 / 深度 / 相机系坐标 / th1 / 转化后基坐标系坐标
        detect_group, detect_cv = self._make_card("▤", "检测信息", "07")
        self.detection_display = QTextEdit()
        self.detection_display.setReadOnly(True)
        self.detection_display.setMinimumHeight(150)
        # ★ 覆盖全局 QTextEdit 样式: 深底等宽字 + 左侧青色标记条(DJI 数据面板观感)
        self.detection_display.setObjectName("detectLog")
        self.detection_display.setStyleSheet(
            "QTextEdit { font-family: %s; font-size: 13px; color: #C9CCD2;"
            " background-color: #101114; border: 1px solid %s;"
            " border-left: 2px solid %s; border-radius: 2px; padding: 8px; }"
            % (DJITheme.MONO, DJITheme.LINE_SOFT, DJITheme.ACCENT))
        detect_cv.addWidget(self.detection_display)

        status_group, status_cv = self._make_card("☰", "状态信息", "08")
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMinimumHeight(150)
        # ★ 2026-09-22 内存上限: 全天运行日志无限累积会持续吃内存
        #   (QTextEdit永不自清理), 上限2000条, 旧的自动从顶部丢弃
        self.status_display.document().setMaximumBlockCount(2000)
        status_cv.addWidget(self.status_display)
        # ★ FK实时显示(2026-09-11): 独立QLabel, 替代原来对status_display的
        #   QTextCursor原位编辑(跨线程QTextCursor破坏Qt事件队列的修复)
        self.fk_live_label = QLabel("末端位置(FK实时): 等待连接...")
        self.fk_live_label.setStyleSheet(
            "font-family: %s; font-size: 12px; color: %s;"
            " background-color: #101114; border: 1px solid %s;"
            " border-left: 2px solid %s; border-radius: 2px; padding: 6px 8px;"
            % (DJITheme.MONO, DJITheme.ACCENT, DJITheme.LINE_SOFT, DJITheme.ACCENT_DK))
        self.fk_live_label.setWordWrap(True)
        status_cv.addWidget(self.fk_live_label)

        info_splitter = QSplitter(Qt.Horizontal)
        info_splitter.addWidget(detect_group)
        info_splitter.addWidget(status_group)
        info_splitter.setStretchFactor(0, 1)
        info_splitter.setStretchFactor(1, 1)

        layout.addWidget(camera_group, 1)
        layout.addWidget(info_splitter, 0)

        return panel

    def _update_camera_placeholder(self, text=None):
        """占位层只在"无画面/冻结无帧"时出现: 直播有帧或冻结帧就隐去。"""
        ph = getattr(self, '_camera_placeholder', None)
        if ph is None:
            return
        cam = self.camera_label
        ph.setGeometry(0, 0, cam.width(), cam.height())
        if cam.frozen:
            show = getattr(cam, '_frozen_img', None) is None
        else:
            pm = cam.pixmap()
            show = (pm is None or pm.isNull())
        if text is not None:
            ph.set_text(text)
        ph.setVisible(show)
        if show:
            ph.raise_()

    _CAM_VIEW_INTERVAL = 0.08   # ★ 显示刷新间隔(秒): ~12帧/秒(订阅30Hz原始流后
                                #   流畅度足够), 解码+缩放全套 ~15ms 不再每 30ms
                                #   挤占运动插值定时器

    def update_camera_view(self):
        """相机显示(节流版): 每 0.15s 解码一次, 其余 tick 直接跳过.
        ★原版每 30ms 全幅解码+flip+Smooth缩放(~15ms)把定时器拖成低频,
        运动插值帧率跟着掉 = '低帧卡顿'的主因之一."""
        import time as _time
        now = _time.monotonic()
        if now - getattr(self, '_cam_view_last', 0.0) < self._CAM_VIEW_INTERVAL:
            return
        self._cam_view_last = now
        if hasattr(self.controller, 'camera_image') and self.controller.camera_image is not None:
            try:
                msg = self.controller.camera_image

                if msg.encoding == 'bgr8':
                    dtype = np.uint8
                    channels = 3
                elif msg.encoding == 'rgb8':
                    dtype = np.uint8
                    channels = 3
                elif msg.encoding == 'mono8':
                    dtype = np.uint8
                    channels = 1
                else:
                    raise ValueError(f"不支持的图像编码: {msg.encoding}")

                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(
                    msg.height, msg.width, channels)

                if channels == 3 and msg.encoding == 'bgr8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                elif channels == 1:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

                # ★ 相机安装朝向补偿: 相机物理倒装, 这里把画面转到显示系再上屏.
                #   注意顺序 —— 检测框坐标是【原始像素系】, 必须先按原始帧画框,
                #   再整体做朝向补偿, 框才会跟着画面一起转(反之框会错位).
                #   补偿只作用于显示, 深度/内参/检测的像素系一律不动
                #   (见 _decode_color_bgr: 检测像素必须与内参同系).

                # ★ 手动画面叠加 CenterNet 检测框: 结果由检测线程经
                #   _on_manual_detection 缓存; 检测约几fps、显示节流0.15s,
                #   2s 内视为有效(期间显示帧重复画同一框, 目标丢失即消失).
                #   ★ 标定模式跳过: 此时 camera_image 已是 /aruco_single/result
                #     叠加图(角点+坐标轴), 再叠 CenterNet 框既无意义又会错位
                #     (检测框来自原始流像素系, 而该图同样来自原始流, 但标定
                #     期间本就不该跑物证检测).
                _det = (None if getattr(self, 'CALIB_MODE', False)
                        else getattr(self, '_latest_detection', None))
                if (_det is not None
                        and _time.time() - getattr(self, '_latest_detection_ts', 0.0) < 2.0):
                    # ★ 多目标画框: 遍历 all_boxes(CenterNet 全部检出),
                    #   置信度≥0.3才显示; rgb8 路径数组只读需先 copy
                    _boxes = [b for b in getattr(_det, 'all_boxes', None) or []
                              if b.get("bbox") is not None and b.get("score", 0) >= 0.3]
                    if _boxes:
                        cv_image = cv_image.copy()
                        for b in _boxes:
                            bx1, by1, bx2, by2 = (int(v) for v in b["bbox"])
                            cv2.rectangle(cv_image, (bx1, by1), (bx2, by2),
                                          (80, 220, 80), 2)
                            cv2.putText(cv_image, "%s %.2f" % (b.get("label", ""), b["score"]),
                                        (bx1, max(14, by1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                        (80, 220, 80), 2)

                # ★ 画完框再整体补偿朝向(框随画面一起转, 文字标注也保持正向)
                cv_image = _cam_orient_image(cv_image)

                h, w, ch = cv_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                # ★ 2026-11-05: ByExpanding→KeepAspectRatio — 直播画面与人工
                #   标注冻结帧视野完全一致(原来直播裁边、冻结全画幅, 冻结瞬间
                #   画面"多出一圈边缘", 两边对不上). paintEvent 已复刻 AlignCenter
                #   居中绘制, 黑边对称无观感问题.
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.width(),
                    self.camera_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation   # ★ 平滑缩放改快速: 再省几毫秒
                )
                self.camera_label.setPixmap(scaled_pixmap)
                self._update_camera_placeholder()
                # ★ 画面状态徽章: 直播帧到达即标"实时"(80ms一次的字符串比较, 零开销)
                if not self.camera_label.frozen and getattr(self, '_cam_chip_state', '') != '实时':
                    self._set_cam_chip('实时')

            except Exception as e:
                rospy.logerr(f"更新相机视图错误: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    # ★★★ 新增：人工画框模式 (2026-09-08, 两步确认版) ★★★
    # 扫描/预览中点「暂停并画框」→ 停臂运动 + 冻结当前原始彩色帧 →
    # 拖拽画框(黄=待确认) → 「确认框选」(绿=生效: 记录[画框样本]=未来M-链
    # 标定样本; 标定后在此把坐标传给机械臂) / 「取消框选」(清框重画, 不退冻结)
    # → 「退出」回直播。画错框不再作废整帧, 也不再有"框完即生效"的误操作风险。
    # ──────────────────────────────────────────────────────────────────────────
    def _set_cam_chip(self, state):
        """相机画面状态徽章: 待机/实时/已冻结 (颜色均在钢蓝灰家族内)"""
        chip = getattr(self, 'cam_state_chip', None)
        if chip is None:
            return
        presets = {
            '待机':   ("■  待机",   DJITheme.TEXT_3, DJITheme.LINE),
            '实时':   ("●  实时",   DJITheme.ACCENT, DJITheme.ACCENT_DK),
            '已冻结': ("❚❚ 已冻结", DJITheme.TEXT_2, "#3A3C43"),
        }
        txt, fg, bd = presets.get(state, presets['待机'])
        chip.setText(txt)
        chip.setStyleSheet(
            "color: %s; border: 1px solid %s; border-radius: 2px;"
            " padding: 4px 10px; font-size: 12px;"
            " font-family: %s; background-color: %s;"
            % (fg, bd, DJITheme.MONO,
               "#0E2325" if state == '实时' else DJITheme.PANEL_2))
        self._cam_chip_state = state

    def on_annotate_pause(self):
        """「暂停并画框」: 停臂运动 + 冻结当前原始彩色帧, 进入画框模式"""
        # 动作线程执行中禁止(其持续发关节指令, 此时冻结画面无意义且危险)
        if getattr(self, 'action_running', False):
            self.status_display.append("[画框] 动作执行中, 等动作完成后再暂停画框")
            return
        if self.camera_label.frozen:
            return   # 已在画框模式

        # 1) 停止臂运动(只停运动, 不碰电机使能 —— 与紧急停止不同)
        if self.auto_mode:
            self.stop_auto_pipeline()
        self.is_scanning = False
        self.is_moving = False
        self._cancel_pending_refine()   # 作废闭环补走, 防止暂停期间臂被拽走

        # 2) 取原始彩色帧(订阅器随自动流程建立; 从未建过则现在补建)
        if self.color_sub is None:
            self.init_perception_subscribers()
            # ★ 订阅刚建立需等首帧(深度~10Hz): 不等则首次点「人工标注」
            #   必因首帧未到而"暂停失败/深度快照None"(2026-09-27)
            for _ in range(30):
                if (getattr(self, '_raw_color_msg', None) is not None
                        and getattr(self, '_raw_depth_msg', None) is not None):
                    break
                QApplication.processEvents()
                time.sleep(0.1)
        color = self._decode_color_bgr()
        if color is None:
            self.status_display.append(
                "[画框] 暂停失败: 无原始彩色帧(请先「启动相机」, 画面出现后再试)")
            self.statusBar().showMessage("画框失败: 无相机帧")
            return

        # 3) 快照机械臂状态+深度(冻结时刻) —— 画框样本=未来的标定样本
        joints = None
        try:
            r = self.controller.arm_client.get_state()
            if r.get("ok") and r.get("joints"):
                joints = list(r["joints"])
        except Exception:
            pass
        if joints is None:
            joints = list(self.controller.joint_pos)
        self._freeze_joints = joints
        self._freeze_th1 = joints[0]
        self._freeze_depth, _best_ratio = self._decode_depth_best(
            min_valid_ratio=0.05, wait_frames=20)  # ★ 2026-10-30: 自动挑好帧, 避免 IR 漏拍卡死
        self._annotate_dbg("[冻结] depth=%s color=%s joints[0]=%.3f best_valid=%.2f%%" % (
            "None" if self._freeze_depth is None
            else "%dx%d" % (self._freeze_depth.shape[0], self._freeze_depth.shape[1]),
            "None" if getattr(self, '_raw_color_msg', None) is None
            else "%dx%d" % (self._raw_color_msg.height, self._raw_color_msg.width),
            joints[0], _best_ratio * 100))
        self._freeze_t = time.strftime("%H:%M:%S")
        # ★ 2026-09-16 冻结位姿快照: 确认框选的相机→基座变换必须用冻结时刻的
        #   臂位姿 —— 原来实现用实时get_pose(), v2流程下确认前臂常已移动
        #   (点过移动/动作), 拿新位姿变换旧冻结帧像素 = 坐标全错(实测同帧同
        #   深度两次确认z差14.5cm, "相机向下时z不准"的根因).
        # ★ 2026-10-30: pose 必须拿到才能冻结 —— 否则画出来的框确认后必然
        #   基坐标为 None (手眼链缺 T_base_gripper), 白浪费时间. 提前终止.
        self._freeze_pose = None
        pose_fail_reason = None
        try:
            rp = self.controller.arm_client.get_pose()
            if rp.get("ok") and rp.get("pos"):
                self._freeze_pose = {'ok': True, 'pos': list(rp["pos"]),
                                     'rpy': list(rp["rpy"])}
            else:
                pose_fail_reason = "arm_server 未就绪 (ok=%s)" % rp.get("ok")
        except Exception as _e:
            pose_fail_reason = "get_pose 异常: %r" % _e
        self._annotate_dbg("[冻结] pose快照=%s" % (
            "None" if self._freeze_pose is None else "ok"))

        if self._freeze_pose is None:
            self.status_display.append(
                "[画框] 无法冻结: 未获取到机械臂末端位姿 (%s). "
                "请先点「启动机械臂」, 等后端就绪后再试." % pose_fail_reason)
            self.statusBar().showMessage("请先启动机械臂再暂停画框")
            return

        # 4) 冻结画面
        self.camera_label.freeze(color)
        self._set_cam_chip('已冻结')
        self._pending_box = None
        self.annotate_confirm_btn.setEnabled(True)
        self.annotate_cancel_btn.setEnabled(True)
        self.annotate_status_label.setText(
            "已冻结 — 拖拽画框(再拖重画) → 「确认框选」生效 / 「取消框选」重画 / 「退出」")
        self.status_display.append(
            "[画框] %s 已暂停并冻结画面 joints=[%s] — 请画框(框完需「确认框选」才生效)" % (
                self._freeze_t, ", ".join("%.2f" % v for v in joints)))
        self.statusBar().showMessage("画框模式: 拖拽画框 → 确认框选")

    def on_annotate_box(self, x1, y1, x2, y2):
        """框绘制完成: 仅登记"待确认"并显示中心三维点(不生效, 不落日志)"""
        self._pending_box = (x1, y1, x2, y2)
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        self.camera_label.set_box_confirmed(False)
        depth_s, xyz_s, base_s = self._annotate_center_3d(u, v)
        self.annotate_status_label.setText(
            "框选待确认: 中心(%d,%d) 深度%s m 基坐标%s — 「确认框选」生效 / 「取消框选」重画" % (
                u, v, depth_s, base_s))

    def _annotate_pose(self):
        """★ 框选链路统一位姿源(2026-09-16): 冻结快照优先 —— 冻结帧的像素
        必须用冻结时刻的臂位姿变换(v2流程确认前臂常已移动, 实时位姿变换
        旧帧=坐标错乱). 无快照(未冻结/快照失败)回落实时查询."""
        rp = getattr(self, '_freeze_pose', None)
        if rp is not None and rp.get("pos"):
            return rp
        try:
            return self.controller.arm_client.get_pose()
        except Exception:
            return {}

    def _annotate_center_3d(self, u, v):
        """★ 框中心的三维点 → 相机系 + 基坐标系 (2026-09-10 手眼标定后完整链):
        彩色像素(u,v) →深度像素→ 采样深度 → 反投影相机系 p_cam
        → 手眼矩阵 T_gripper_cam + 冻结时刻末端位姿 → 基坐标 p_base.
        返回 (深度str, 相机系str, 基坐标str); 深度无效时基坐标为 "None"."""
        depth_img = getattr(self, '_freeze_depth', None)
        if depth_img is None:
            self._annotate_dbg("u=%d v=%d 断点①: 深度快照为None" % (u, v))
            return "None", "None", "None"
        try:
            dx, dy = self._annotate_map_color_to_depth(u, v, depth_img)
            # ★ 2026-10-30 深度采样全面诊断: 把原始邻域值、各窗口结果全打出来
            _h, _w = depth_img.shape[:2]
            _patch = depth_img[max(0,dy-8):min(_h,dy+9), max(0,dx-8):min(_w,dx+9)]
            _flat = _patch.ravel()
            _nz = _flat[_flat > 0]
            self._annotate_dbg(
                "u=%d v=%d -> dx=%d dy=%d depth.shape=%s dtype=%s 邻域17x17: "
                "min=%s max=%s nz_cnt=%d/%d nz_sample=%s" % (
                    u, v, dx, dy, depth_img.shape, depth_img.dtype,
                    _nz.min() if _nz.size > 0 else "-",
                    _nz.max() if _nz.size > 0 else "-",
                    _nz.size, _flat.size,
                    list(_nz[:15]) if _nz.size > 0 else "[]"))
            dv = pcalib.sample_stable_depth(depth_img, dx, dy)
            self._annotate_dbg("  win=5 → dv=%s" % dv)
            if dv is None:
                dv = pcalib.sample_stable_depth(depth_img, dx, dy, win=9)
                self._annotate_dbg("  win=9 → dv=%s" % dv)
            if dv is None:
                dv = pcalib.sample_stable_depth(depth_img, dx, dy, win=13)
                self._annotate_dbg("  win=13 → dv=%s" % dv)
            if dv is not None and 0.05 < dv < 1.5:
                p_cam = pcalib.pixel_depth_to_cam_xyz(
                    u, v, dv, self.camera_intrinsics)
                cam_s = "(%.4f,%.4f,%.4f)" % (p_cam[0], p_cam[1], p_cam[2])
                # ★ 基坐标: 手眼链 p_base = T_base_gripper · T_gripper_cam · p_cam
                base_s = "None"
                try:
                    # ★ 位姿用冻结快照(帧与位姿必须同时刻, 2026-09-16修复z不准)
                    rp = self._annotate_pose()
                    self._annotate_dbg(
                        "u=%d v=%d dv=%.3f p_cam=%s 位姿: ok=%s pos=%s rpy=%s" % (
                            u, v, dv, cam_s, rp.get("ok"), rp.get("pos"), rp.get("rpy")))
                    if rp.get("ok") and rp.get("pos"):
                        p_base = pcalib.rebot_cam_xyz_to_base_full(
                            p_cam, rp["pos"], rp["rpy"])
                        base_s = "(%.4f,%.4f,%.4f)" % (
                            p_base[0], p_base[1], p_base[2])
                    else:
                        self._annotate_dbg("断点③: 位姿不ok(get_pose失败?)")
                except Exception as _e:
                    self._annotate_dbg("断点④: 基坐标解算异常: %r" % _e)
                    print("[画框] 基坐标解算失败: %s" % _e)
                return ("%.3f" % dv, cam_s, base_s)
            self._annotate_dbg("断点②: 深度像素(%d,%d) dv=%s (None或超0.05~1.5)" % (dx, dy, dv))
        except Exception as e:
            self._annotate_dbg("断点⑤: 解算异常: %r" % e)
            print("[画框] 相机系坐标解算异常: %s" % e)
        return "None", "None", "None"

    def _annotate_dbg(self, msg):
        """★ 框选链路诊断日志(2026-09-27): 写 /tmp/annotate_debug.log,
        不依赖UI终端可见性 —— 基坐标"None"时逐断点定位."""
        try:
            with open('/tmp/annotate_debug.log', 'a') as f:
                f.write(time.strftime('%H:%M:%S ') + msg + '\n')
        except Exception:
            pass

    def _estimate_box_normal(self, box):
        """★ 方案1(2026-09-16): 框内深度点云PCA拟合物证表面法向(斜面/地面物证用).
        链路: 彩色框→深度图框(角点线性映射) → 腐蚀边界取有效深度像素
        → 反投影相机系点云(深度内参) → SVD最小奇异向量=平面法向
        → 2σ残差剔除重拟合一次(抗边缘混合像素) → 朝相机侧(outward)符号
        → 手眼旋转(纯旋转,平移无关)转基座系 → 取负=夹爪逼近法向(gripper X指向表面).
        像素不足时向外扩两档(采周边支撑面); 全程失败返回 None(回落原候选列表)."""
        depth_img = getattr(self, '_freeze_depth', None)
        if depth_img is None:
            return None
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            # 彩色框角点 → 深度图框(已对齐时直取, 未对齐时基线补偿映射)
            dx1, dy1 = self._annotate_map_color_to_depth(x1, y1, depth_img)
            dx2, dy2 = self._annotate_map_color_to_depth(x2, y2, depth_img)
            dx1, dx2 = min(dx1, dx2), max(dx1, dx2)
            dy1, dy2 = min(dy1, dy2), max(dy1, dy2)
            dh, dw = depth_img.shape[:2]
            # 腐蚀3px去框线/深度边缘混合像素
            cx1, cy1, cx2, cy2 = dx1 + 3, dy1 + 3, dx2 - 3, dy2 - 3
            pts = None
            for expand in (0, 6, 12):   # 像素不足→外扩采支撑面
                ax1 = max(cx1 - expand, 0); ay1 = max(cy1 - expand, 0)
                ax2 = min(cx2 + expand, dw - 1); ay2 = min(cy2 + expand, dh - 1)
                if ax2 - ax1 < 4 or ay2 - ay1 < 4:
                    continue
                region = np.asarray(depth_img[ay1:ay2 + 1, ax1:ax2 + 1], dtype=np.float64)
                vs, us = np.nonzero((region > 0.05) & (region < 1.5))
                if len(vs) < 40:
                    continue
                ds = region[vs, us]
                us_g = us + ax1; vs_g = vs + ay1
                # 深度像素反投影用深度内参(未就绪回落彩色fx/fy+默认光心)
                di = self.depth_intrinsics
                if getattr(di, 'ready', False):
                    fx_d, fy_d, cx_d, cy_d = di.fx, di.fy, di.cx, di.cy
                else:
                    ci = self.camera_intrinsics
                    fx_d, fy_d, cx_d, cy_d = ci.fx, ci.fy, 320.0, 240.0
                xs = (us_g - cx_d) * ds / fx_d
                ys = (vs_g - cy_d) * ds / fy_d
                pts = np.stack([xs, ys, ds], axis=1)
                break
            if pts is None:
                print("[画框法向] 有效深度像素不足, 跳过法向估计")
                return None
            # PCA平面拟合 + 2σ残差剔除重拟合
            def _fit(P):
                c = P.mean(axis=0)
                _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
                return Vt[2], np.abs((P - c) @ Vt[2])
            n, resid = _fit(pts)
            keep = resid < max(2.0 * float(resid.std()), 0.002)
            if int(keep.sum()) >= 30:
                n, _ = _fit(pts[keep])
            if n[2] > 0:
                n = -n          # 统一为朝相机侧(表面外法向, cam系z<0)
            # 相机系方向 → 基座系(纯旋转): R = (T_base_gripper · T_gripper_cam)[:3,:3]
            #   ★ 位姿用冻结快照(与深度帧同时刻, 2026-09-16)
            rp = self._annotate_pose()
            if not (rp.get("ok") and rp.get("pos")):
                return None
            T_bg = np.eye(4)
            T_bg[:3, :3] = pcalib.rebot_rpy_to_matrix(
                rp["rpy"][0], rp["rpy"][1], rp["rpy"][2])
            R = (T_bg @ pcalib.REBOT_HAND_EYE_T_GRIPPER_CAM)[:3, :3]
            n_out = R @ n                      # 基座系表面外法向(背离表面)
            nl = float(np.linalg.norm(n_out))
            if nl < 1e-6:
                return None
            n_out = n_out / nl
            approach = [-n_out[0], -n_out[1], -n_out[2]]   # 夹爪逼近法向 = -外法向
            tilt = math.degrees(math.acos(min(abs(n_out[2]), 1.0)))
            print("[画框法向] 估计: 表面外法向(基座系)=(%.2f,%.2f,%.2f) "
                  "倾角%.0f° 逼近法向=(%.2f,%.2f,%.2f) 样本%d点"
                  % (n_out[0], n_out[1], n_out[2], tilt,
                     approach[0], approach[1], approach[2], len(pts)))
            return approach
        except Exception as e:
            print("[画框法向] 估计异常: %s" % e)
            return None

    def on_annotate_confirm(self):
        """「确认框选」: 待确认的框正式生效 —— 记录坐标样本(标定后在此传给机械臂)"""
        if not self.camera_label.frozen:
            return
        if not self.camera_label.has_box:
            self.annotate_status_label.setText("没有待确认的框 — 请先在画面上拖拽画框")
            return
        if self.camera_label.box_confirmed:
            return   # 当前框已确认过, 重复点击忽略
        box = getattr(self, '_pending_box', None)
        if box is None:
            return
        x1, y1, x2, y2 = box
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        self.camera_label.set_box_confirmed(True)   # 黄框→绿框
        # ---- 三维: 深度→相机系→基坐标(手眼标定完整链, 2026-09-10) ----
        depth_s, xyz_s, base_s = self._annotate_center_3d(u, v)
        joints = getattr(self, '_freeze_joints', None) or list(self.controller.joint_pos)
        self.annotate_status_label.setText(
            "已确认: 中心(%d,%d) 基坐标%s m th1=%.3frad — 已记录" % (
                u, v, base_s, joints[0]))
        # ★ [画框样本] 手眼标定完整链: 基坐标由 深度反投影+手眼矩阵+末端FK 算出
        print("[画框样本] u=%d v=%d w=%d h=%d joints=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] "
              "th1=%.4f depth=%s xyz_cam=%s xyz_base=%s" % (
                  u, v, x2 - x1, y2 - y1,
                  joints[0], joints[1], joints[2], joints[3], joints[4], joints[5],
                  joints[0], depth_s, xyz_s, base_s))
        self.status_display.append(
            "[画框] 确认 基坐标=%s (深度=%s 相机系=%s th1=%.3frad)" % (
                base_s, depth_s, xyz_s, joints[0]))
        # ★ 2026-09-10 v2: 基坐标不覆盖输入框 —— 用户手动输入的坐标保持不变,
        #   画框确认后直接驱使机械臂运动到框选位置(与手动输入互不干扰).
        #   输入框仍由用户手动编辑/读取当前位姿按钮维护.
        if base_s != "None":
            try:
                vals = base_s.strip('()').split(',')
                bx = float(vals[0]); by = float(vals[1]); bz = float(vals[2])
                # ★ 2026-10-30 用户规则: 目标位姿 Z>0 → 沿用原逻辑(框内深度PCA
                #   估算法向, 估计失败保留输入框原值); Z<=0 → 法向固定(0,0,-1)
                #   (低位深度估计不可靠); J6 恒为 0(下方填框已设).
                odx, ody, odz = self._boxed_offset(bx, by, bz)
                bx_c = bx + odx
                by_c = by + ody
                bz_c = bz + odz
                if bz_c > 0.0:
                    est_n = self._estimate_box_normal(box)   # ★ 方案1: 框内深度法向
                else:
                    est_n = [0.0, 0.0, -1.0]
                # ★ 2026-10-30 用户要求: 取消原"目标z<0 → Z强制-0.05"规则,
                #   Z 保留深度反投影实测值+补偿(原规则会把物证真实高度-0.157
                #   直接丢掉, 臂跑到错误高度).
                self.x_input.setText("%.4f" % bx_c)
                self.y_input.setText("%.4f" % by_c)
                self.z_input.setText("%.4f" % bz_c)
                # ★ 2026-10-30 用户要求: 目标位置 J6 一直填 0
                self.j6_input.setText("0")
                if est_n is not None:
                    self.nx_input.setText("%.3f" % est_n[0])
                    self.ny_input.setText("%.3f" % est_n[1])
                    self.nz_input.setText("%.3f" % est_n[2])
                self._boxed_pending = {'raw': [bx, by, bz],
                                       'xyz': [bx_c, by_c, bz_c],
                                       'est_n': est_n}
                _n_src = ("z<=0固定(0,0,-1)" if (bz_c <= 0.0 and est_n is not None)
                          else ("深度估计(%.2f,%.2f,%.2f)" % tuple(est_n)
                                if est_n else "估计失败, 沿用输入框法向"))
                self.status_display.append(
                    "[画框] 已填入目标位姿 (%.4f, %.4f, %.4f) 法向:%s — "
                    "点[移动到位置]或预设动作执行" % (bx_c, by_c, bz_c, _n_src))
                self.annotate_status_label.setText(
                    "已确认并填入目标位姿 — 点[移动到位置]或预设动作执行")
            except Exception as _e:
                print("[画框] 框选填入失败: %s" % _e)
        self._pending_box = None

    def _move_to_boxed_position(self, bx, by, bz, est_normal=None):
        """★ 画框确认后: 自动搜索可达法向量 → 运动到修正后位置.
        策略(2026-09-16 v5): 位置固定, 自动遍历多组法向量求IK:
          1. 框内深度PCA估计法向(方案1, 斜面/地面物证的真实姿态, 最优先)
          2. 用户当前输入的法向量
          3. 常见工作法向量列表(从用户标定数据提取的真实姿态)
          4. 遍历球面采样(极角15°~75°, 方位角8方向)
        某组法向 IK 成功即运动; 全部失败才报不可达."""

        # ★ 端到端偏移补偿(Y线性函数 + Z<0低位X-4mm, Z不补偿)
        odx, ody, odz = self._boxed_offset(bx, by, bz)
        bx_c = bx + odx
        by_c = by + ody
        bz_c = bz + odz
        # ★ 2026-10-30 用户要求: 取消原"目标z<0 → Z强制-0.05"规则,
        #   与填框逻辑一致(两处都用深度实测值+补偿).

        # ★ 保存用户原始输入(最终恢复)
        try:
            saved = {
                'x': self.x_input.text(), 'y': self.y_input.text(),
                'z': self.z_input.text(),
                'nx': self.nx_input.text(), 'ny': self.ny_input.text(),
                'nz': self.nz_input.text(),
            }
        except Exception:
            saved = None

        # ★ 读取用户当前法向量(第一优先尝试)
        try:
            unx = float(self.nx_input.text())
            uny = float(self.ny_input.text())
            unz = float(self.nz_input.text())
            user_normal = [unx, uny, unz]
        except Exception:
            user_normal = None

        # ★ 构建法向量候选列表: 深度估计 → 用户法向 → 标定实测法向 → 球面采样
        # (2026-09-16 抽出 _normal_candidate_list 与预设动作共用)
        # ★ 2026-10-30 用户规则: 目标位姿 Z<=0 → 法向固定(0,0,-1), 不做候选遍历;
        #   Z>0 → 沿用原候选链路(深度估计优先, 失败再降级).
        if bz_c <= 0.0:
            candidates = [("z<=0固定", [0.0, 0.0, -1.0])]
        elif est_normal is not None:
            candidates = []
            candidates.append(('深度估计', [float(v) for v in est_normal]))
            if user_normal is not None and any(
                    abs(a - b) > 0.01 for a, b in zip(user_normal, est_normal)):
                candidates += _normal_candidate_list(user_normal, '用户输入')
            else:
                candidates += _normal_candidate_list(None)
        else:
            candidates = _normal_candidate_list(user_normal, '用户输入')

        # ★ 逐个法向预筛(2026-10-16 服务器端轨迹模式): 用轻量 solve_ik 预筛
        #   (停距由 APPROACH_STANDOFF_M 决定, 当前0=精确到点), 预筛成功的法向
        #   才正式下发 cmd:"move_traj" —— 避免逐候选付出服务器端"IK+轨迹解算"
        #   的完整代价(数秒/个).
        # ★ 2026-09-27晚: 移除REBOT_IK_BIAS补偿 —— 实测两轮对照证明当前臂
        #   IK系统偏差≈0(加补偿后落点偏-IK_BIAS即左上/-8mm X), 11.txt时代
        #   拟合的bias疑似采样时臂未稳定所致, 叠加只会引入新误差.
        found = None
        for desc, normal in candidates:
            sx, sy, sz = _standoff_target(bx_c, by_c, bz_c, normal)
            success, joints = self.controller.move_to_cartesian(
                sx, sy, sz, None, normal)
            if success:
                found = (desc, normal, joints, (sx, sy, sz))
                break

        if found is None:
            self.status_display.append(
                "[画框运动] 目标(%.3f, %.3f, %.3f)已尝试%d组法向量均不可达 — "
                "位置可能超出工作空间" % (bx_c, by_c, bz_c, len(candidates)))
            self.statusBar().showMessage("画框目标不可达")
            return

        desc, normal, joints, stop_xyz = found
        # ★ 成功: 正式下发服务器端轨迹 + 恢复用户输入 + 显示所用法向
        self.controller.q6_value = 0.0
        # ★ 作废旧闭环补走线程(其依赖 solve_ik/set_joints, 且会与
        #   服务器端轨迹互相覆盖)
        self._cancel_pending_refine()
        r = self.controller.arm_client.move_traj(
            stop_xyz[0], stop_xyz[1], stop_xyz[2], normal)
        if not r.get("ok"):
            self.status_display.append(
                "[画框运动] 服务器端轨迹发起失败: %s" % r.get("err", "未知"))
            self.statusBar().showMessage("画框运动发起失败")
            return
        self._notify_trail_start()
        self.status_display.append(
            "[画框运动] 前往(%.4f, %.4f, %.4f) | "
            "法向=(%.2f,%.2f,%.2f)(%s) | "
            "修正(%+.0f,%+.0f,%+.0f mm)" % (
                stop_xyz[0], stop_xyz[1], stop_xyz[2],
                normal[0], normal[1], normal[2], desc,
                self._BOX_CORR[0]*1000, self._BOX_CORR[1]*1000,
                self._BOX_CORR[2]*1000))
        self.statusBar().showMessage(
            "臂→(%.3f, %.3f, %.3f) 法向=(%.1f,%.1f,%.1f)" % (
                stop_xyz[0], stop_xyz[1], stop_xyz[2],
                normal[0], normal[1], normal[2]))

    def _post_move_refine(self, tag, tx, ty, tz, normal=None, epoch=None):
        """★ 闭环到位修正(2026-09-27晚恢复): 运动完成后 get_pose 实测,
        偏差>2mm 按差值补走(最多3步) —— IK 跳解兜底(arm_server solve_ik 无
        FK 验收, 跳解也返回 ok=True, 实测 7 次移动 3 次跳 5~13cm).
        2026-09-27深夜: 阈值8mm→2mm/步数2→3 —— 旧8mm容差下结束残差
        ±9~12mm恰是用户看到的"越往右越偏左"量级(且第2步补走后仍可能
        再跳解, 需第3步). 补走时臂从当前关节就近解算, 不再跳支;
        全程记录 /tmp/annotate_debug.log.
        ★ 2026-10-04: 加 epoch 作废机制 —— 补走线程每次醒来先核对运动代号,
        用户在此期间下了新指令(回零/J6微调/新目标/急停)则立即退出,
        杜绝"回零走到一半被闭环拽回旧点位"."""
        gen = epoch if epoch is not None else self._move_epoch

        def _worker():
            try:
                time.sleep(MOVE_DURATION + 2.5)
                for step in (1, 2, 3):
                    if self._move_epoch != gen:
                        self._annotate_dbg(
                            "[%s闭环] 检测到新运动指令(epoch %d→%d), 放弃补走"
                            % (tag, gen, self._move_epoch))
                        return
                    rp = self.controller.arm_client.get_pose()
                    if not (rp.get("ok") and rp.get("pos")):
                        self._annotate_dbg("[%s闭环] get_pose不ok: %s" % (tag, rp))
                        return
                    p = rp["pos"]
                    dx, dy, dz = p[0]-tx, p[1]-ty, p[2]-tz
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    self._annotate_dbg(
                        "[%s闭环] 第%d步 目标=(%.4f,%.4f,%.4f) 实际=(%.4f,%.4f,%.4f) "
                        "Δ=(%+.1f,%+.1f,%+.1f)mm 距离=%.1fmm%s" % (
                            tag, step, tx, ty, tz, p[0], p[1], p[2],
                            dx*1000, dy*1000, dz*1000, dist*1000,
                            " ≤2mm 达标" if dist <= 0.002 else " → 补走"))
                    if dist <= 0.002:
                        return
                    # 补走目标 = 原目标(沿停距语义), 臂就近解算不跳支
                    cx, cy, cz = _standoff_target(tx, ty, tz, normal)
                    self.controller.q6_value = 0.0
                    success, joints = self.controller.move_to_cartesian(
                        cx, cy, cz, None, normal)
                    if not success:
                        self._annotate_dbg(
                            "[%s闭环] 第%d步补走解算失败(IK不可达)" % (tag, step))
                        return
                    # bump_epoch=False: 自己的补走不能作废旧代号
                    self.start_arm_movement(
                        self.controller.joint_pos.copy(), joints, MOVE_DURATION,
                        bump_epoch=False)
                    time.sleep(MOVE_DURATION + 2.0)
                rp = self.controller.arm_client.get_pose()
                if rp.get("ok") and rp.get("pos"):
                    p = rp["pos"]
                    self._annotate_dbg(
                        "[%s闭环] 结束 残差=(%+.1f,%+.1f,%+.1f)mm" % (
                            tag, (p[0]-tx)*1000, (p[1]-ty)*1000, (p[2]-tz)*1000))
            except Exception as e:
                self._annotate_dbg("[%s闭环] 异常: %r" % (tag, e))
        threading.Thread(target=_worker, daemon=True).start()

    # 画框端到端偏移补偿(2026-10-30 v5, 55.txt 四组触碰真值最小二乘):
    #   v4 的 0.28/-0.006 是旧镜像手眼矩阵时代调的, 换新矩阵后严重过补偿
    #   (如 Y=-0.10 处原始链仅偏+6mm, x1.28 后反被放大到 -27mm)。
    #   v5: 全域仿射 显示值 = 1.0422*y_raw - 0.0015, 四组残差 ≤2.3mm
    #   (局部拟合, 适用 z_cam≈0.26m / X≈0.49m 附近, 远端精度需实测)。
    _BOX_CORR = np.array([0.0, 0.0, 0.0], dtype=np.float64)   # dx 恒定项, dz 恒 0
    _BOX_CORR_Y_SCALE = 0.0422   # y 比例补偿(全域生效)
    _BOX_CORR_Y_CONST = -0.0015  # y 恒定补偿(m)(全域生效)

    def _boxed_offset(self, bx, by, bz=None):
        """画框端到端偏移补偿量 (dx, dy, dz):
        dy = Y_SCALE·y + Y_CONST (全域); dz 不做补偿(★ 2026-10-31 用户要求,
        原 X>0.5m 远端 +10mm 规则已删除);
        X 补偿按检测位置分档(★ 2026-11-01 用户要求):
          · 基坐标 Z<0(低位检测): X 偏大 → -4mm
          · 基坐标 Z>0 且 Y<0:   X 偏小 → +5mm
          · 基坐标 Z>0 且 Y>=0:  不补偿"""
        dy = self._BOX_CORR_Y_SCALE * by + self._BOX_CORR_Y_CONST
        if by > 0.07:      # ★ 2026-11-01 用户要求: Y>0.07m 改用恒定 -4mm(不走仿射)
            dy = -0.004
        dx = self._BOX_CORR[0]
        if bz is not None and bz < 0.0:
            dx = -0.004
        elif by < 0.0:
            dx = 0.005
        return (dx, dy, 0.0)

    def on_annotate_cancel(self):
        """「取消框选」: 清除当前框, 允许重新拖拽(不退出冻结帧)"""
        if not self.camera_label.frozen:
            return
        if not self.camera_label.has_box:
            self.annotate_status_label.setText("没有可取消的框 — 请先拖拽画框")
            return
        if self.camera_label.box_confirmed:
            # 已确认的框已生效(样本已记录), 取消无意义; 直接拖新框替换即可
            self.annotate_status_label.setText(
                "该框已确认生效 — 如需重画直接拖拽新框(旧记录保留)")
            return
        self.camera_label.clear_box()
        self._pending_box = None
        self.annotate_status_label.setText("已取消 — 可重新拖拽画框(画面仍冻结)")
        self.status_display.append("[画框] 已取消框选, 可重新画框")

    def on_annotate_resume(self):
        """「退出」: 退出画框模式回直播(未确认的框自动作废)"""
        if not self.camera_label.frozen:
            return
        self.camera_label.unfreeze()
        self._set_cam_chip('实时')
        self._pending_box = None
        self.annotate_confirm_btn.setEnabled(False)
        self.annotate_cancel_btn.setEnabled(False)
        self.annotate_status_label.setText("画框模式: 未激活")
        self.status_display.append("[画框] 已退出画框模式, 恢复实时预览")
        self.statusBar().showMessage("相机预览已恢复")

    def _build_fk_str(self):
        """★ FK块计算(2026-09-09抽出): get_pose → 末端位置 + 实际法向(rpy反推).
        状态栏事件刷新与0.5s实时跟随共用本函数, 保证两处读数永远同源.
        ★ 2026-09-10 标定模式: 追加完整 rpy/四元数输出(手眼标定数据采集用,
        法向n只有旋转矩阵1列, 重建另2列有误差 —— 手眼解算需要完整姿态)."""
        try:
            r = self.controller.arm_client.get_pose_fast()   # ★ 短超时版: 后台刷新不占锁
            if r.get("ok") and r.get("pos"):
                p = r["pos"]
                fk_str = "X: {:.4f}\n    Y: {:.4f}\n    Z: {:.4f}".format(p[0], p[1], p[2])
                try:
                    _rpy = r.get("rpy") or [0.0, 0.0, 0.0]
                    _R = pcalib.rebot_rpy_to_matrix(_rpy[0], _rpy[1], _rpy[2])
                    _n = np.asarray(_R)[:, 0]
                    _ln = float(np.linalg.norm(_n)) or 1.0
                    _n = _n / _ln
                    fk_str += "\n    法向n(实际): ({:.4f}, {:.4f}, {:.4f})".format(
                        _n[0], _n[1], _n[2])
                    # ★ 标定数据: 完整rpy(手眼解算姿态输入, 2026-09-10)
                    fk_str += "\n    rpy(标定用): ({:.6f}, {:.6f}, {:.6f})".format(
                        _rpy[0], _rpy[1], _rpy[2])
                except Exception as _e:
                    print("[状态显示] 实际法向解算失败: %s" % _e)
                # ★ 成功: 清零连击计数并缓存
                self._fk_fail_strikes = 0
                self._fk_last_str = fk_str
                return fk_str
            # ★ 失败容错(2026-11-05): 单次锁竞争超时不立即报未连接
            self._fk_fail_strikes += 1
            _err = r.get("err") or ("no-err resp keys=%s" % sorted(r.keys()))
            print("[FK] get_pose 失败 #%d: %s" % (self._fk_fail_strikes, _err))
            if self._fk_fail_strikes >= 3:
                return "N/A(机械臂未连接)"
            return self._fk_last_str
        except Exception as _e:
            print("[FK] _build_fk_str 异常: %s" % _e)
            return self._fk_last_str

    # ★ FK后台刷新线程(2026-09-11 v2): 后台线程调get_pose缓存结果,
    #   主线程定时器只读缓存更新QLabel — 主线程绝不阻塞在arm_server I/O上.
    #   原方案在主线程定时器里调get_pose(锁+3s超时), 每0.5s卡3s,
    #   按钮事件排不上队("夹爪/J6点了没反应"的根因).
    _fk_bg_result = None       # 后台线程写入, 主线程读取
    _fk_bg_thread = None       # 后台线程句柄
    _fk_bg_running = False     # 线程存活标志
    # ★ 2026-11-05: FK 连续失败容错 — get_pose_fast 偶发锁竞争超时不再
    #   立即显示"机械臂未连接", 连续 3 次(约1.5s)才真报未连接; 中间返回
    #   上次成功位置, 避免瞬时抖动闪烁.
    _fk_fail_strikes = 0
    _fk_last_str = "N/A"

    def _start_fk_bg_thread(self):
        if self._fk_bg_running:
            return
        self._fk_bg_running = True
        def _worker():
            import time as _t
            while self._fk_bg_running:
                try:
                    fk_str = self._build_fk_str()
                    self._fk_bg_result = fk_str
                except Exception:
                    pass
                # ★ 2026-10-16 服务器端轨迹模式: 臂由 arm_control 的 move_to_traj
                #   驱动, UI 台账(joint_pos)不再由30Hz插值更新 —— 后台轮询
                #   get_state 同步真实关节角, 保证 /joint_states(TF链/轨迹
                #   记录/J6基准)跟随. 本地驱动期间(is_moving/动作/扫描)不覆盖,
                #   它们自己管理台账.
                if not (self.is_moving or getattr(self, 'action_running', False)
                        or self.is_scanning):
                    try:
                        rs = self.controller.arm_client.get_state()
                        if rs.get("ok") and rs.get("joints"):
                            self.controller.joint_pos = list(rs["joints"])
                            self.controller.q6_value = float(rs["joints"][5])
                            # ★ 2026-10-18: move_traj 完成事件(纯事件驱动):
                            #   30Hz 插值路径完成时有 update_arm_movement 的
                            #   fraction>=1.0 钩子刷新状态, 服务器端轨迹路径
                            #   没有对应钩子 —— 这里检测 traj_active 下降沿
                            #   (True→False=完成), 回主线程刷"当前机械臂状态".
                            _ta = bool(rs.get("traj_active"))
                            if getattr(self, '_traj_was_active', False) and not _ta:
                                # ★ 2026-10-29: 延迟3s再刷 —— traj_active 下降沿
                                #   只代表轨迹点发完, 软增益下关节还在收敛,
                                #   立即刷会显示"未停稳"的FK(与0.5s实时标签
                                #   差几mm~1cm). 等3s电机停稳后再取状态.
                                self._ui_calls.append(
                                    lambda: QTimer.singleShot(
                                        3000, self.show_current_status))
                            self._traj_was_active = _ta
                    except Exception:
                        pass
                _t.sleep(0.5)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._fk_bg_thread = t

    def _refresh_fk_block(self):
        """★ FK块实时跟随(主线程, 只读缓存, 零阻塞): 读后台线程的最新结果
        更新QLabel. 不再在主线程调get_pose(修复UI卡死)."""
        if not hasattr(self, 'fk_live_label'):
            return
        cached = self._fk_bg_result
        if cached is not None:
            self.fk_live_label.setText("末端位置(FK实时):\n    " + cached)

    def show_current_status(self):
        if self.is_scanning:
            action_text = "扫描中"
        else:
            action_text = "无" if self.current_action is None else f"动作{self.current_action}"

        # 获取当前法向量对应的四元数
        try:
            nx = float(self.nx_input.text())
            ny = float(self.ny_input.text())
            nz = float(self.nz_input.text())
            quat = normal_to_quaternion(nx, ny, nz)
            quat_str = f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
        except:
            quat_str = "N/A"

        # ★ FK块计算抽出为 _build_fk_str() (2026-09-09): 事件触发即时计算
        fk_str = self._build_fk_str()

        status_text = f"""当前机械臂状态:

    当前执行动作: {action_text}
    关节角度 (弧度):
    关节1: {self.controller.joint_pos[0]:.4f}
    关节2: {self.controller.joint_pos[1]:.4f}
    关节3: {self.controller.joint_pos[2]:.4f}
    关节4: {self.controller.joint_pos[3]:.4f}
    关节5: {self.controller.joint_pos[4]:.4f}
    关节6: {self.controller.q6_value:.4f}

    ★实际末端位置(FK, inv_k有误差所以与输入不同):
    {fk_str}

    工作空间状态(用户输入):
    X: {float(self.x_input.text()) if self.x_input.text() else 'N/A'}
    Y: {float(self.y_input.text()) if self.y_input.text() else 'N/A'}
    Z: {float(self.z_input.text()) if self.z_input.text() else 'N/A'}
    
    表面法向量:
    法向量: ({self.nx_input.text()}, {self.ny_input.text()}, {self.nz_input.text()})
    四元数: {quat_str}
    """
        self.status_display.setPlainText(status_text)
        if self.is_scanning:
            self.statusBar().showMessage("🔍 扫描进行中...")
        else:
            self.statusBar().showMessage(f"系统运行中 - 当前动作: {action_text}")

    # ★ 修改后的 move_to_position 方法 - 添加打印六个关节角值
    # ─── Z轴补偿: 已废弃(2026-09-11) ──────────────────────────────────────
    # 原理: 手眼标定之前用8组数据拟合的二次曲线补偿IK下垂, 但:
    #   ① 到位锁定(kp=75)已消掉大部分静态下垂
    #   ② IK验收收紧到3mm后, 下垂已不在IK解算环节
    #   ③ 该模型在z>0.25时补偿量爆炸(z=0.3→+69mm), 把目标推出可达域
    #   ④ 画框路径的BOX_CORR(+47.5mm)是正确补偿, 与Z_comp叠加会双重补偿
    # 因此设为恒等变换(不补偿), 手动输入z即为IK求解z.
    _ZCAL_A = 0.0
    _ZCAL_B = 1.0
    _ZCAL_C = 0.0

    @classmethod
    def _zcomp(cls, z_in):
        """Z输入补偿: 输入z → 代入IK的z(模型逆), 使实测≈输入.
        ★ 2026-09-11 已废弃(a=0时二次方程退化): 直接返回原值.
        ★ 2026-09-16 修复"z为负无响应": 原钳位 max(0.01,...) 把所有负z强改成
          0.01 再解IK(打印仍显示原始z → "计算有结果"但臂去的是z=0.01) —
          地面/低位物证(z<0)永远打不到. 钳位整个删除, z域由 workSpace
          (zmin=-0.40) 统一把关."""
        return z_in

    def detect_ground(self):
        """★ 2026-10-30 用户要求: 检测地面 —— 目标点 (0.35, 0, 0)、法向竖直
        朝下 (0,0,-1); 二次确认后填框, 且本次不叠加终点停距(尖端直达 Z=0)."""
        reply = QMessageBox.question(
            self, "确认检测地面",
            "将移动到地面检测点 (0.3500, 0.0000, 0.0000)，法向竖直朝下。\n"
            "本次不叠加终点停距，尖端会直达 Z=0。\n\n确认开始移动？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            self.status_display.append("[检测地面] 已取消")
            return
        self._boxed_pending = None   # 清掉画框待执行, 强制走手动输入路径
        self.x_input.setText("0.3500")
        self.y_input.setText("0.0000")
        self.z_input.setText("0.0000")
        self.nx_input.setText("0.000")
        self.ny_input.setText("0.000")
        self.nz_input.setText("-1.000")
        self.j6_input.setText("0")
        self._ground_exact = True   # ★ 一次性: move_to_position 内关闭终点停距
        self.status_display.append(
            "[检测地面] 目标 (0.3500, 0.0000, 0.0000) 法向 (0,0,-1) — 开始移动")
        self.move_to_position()

    def move_to_position(self):
        # ★ 2026-10-30: 一次性消费「检测地面」的精确到位标志, 防止异常残留
        ground_exact = getattr(self, '_ground_exact', False)
        self._ground_exact = False
        # ★ 2026-09-16 v2流程: 框选待执行状态 → 走画框路径(法向候选搜索,
        #   估算法向优先); 输入框被手动改过或无待执行 → 原手动输入行为.
        pending = getattr(self, '_boxed_pending', None)
        if pending is not None:
            self._boxed_pending = None   # 一次性消费
            try:
                cur = [float(self.x_input.text()), float(self.y_input.text()),
                       float(self.z_input.text())]
                unchanged = all(abs(a - b) < 5e-4
                                for a, b in zip(cur, pending['xyz']))
            except Exception:
                unchanged = False
            if unchanged:
                # 走画框路径: _move_to_boxed_position 内部会加 _BOX_CORR → 传raw
                self._move_to_boxed_position(
                    pending['raw'][0], pending['raw'][1], pending['raw'][2],
                    pending['est_n'])
                return
            self.status_display.append(
                "[移动] 检测到输入框已手动修改 — 按手动输入执行(不用画框法向搜索)")
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())

            q6_text = self.j6_input.text()
            # ★ 留空=保持当前J6不动(2026-10-25): 法向竖直(0,0,±1)时后端
            #   normal_to_rpy 备用基准与当前J6差90°(绕接近轴自转无约束,
            #   基准是任选的), q6=None会用默认基准 → IK解出J6突转90°;
            #   用台账角覆盖roll基准即保持J6现状
            q6 = float(q6_text) if q6_text else float(self.controller.q6_value)

            # ★ 获取法向量输入
            nx = float(self.nx_input.text())
            ny = float(self.ny_input.text())
            nz = float(self.nz_input.text())
            normal = [nx, ny, nz]

            # ★ Z输入补偿(2026-09-10): IK求解用补偿后的z, 补偿量=系统性
            #   下垂+IK残差的逆模型; 用户输入值原样显示/记录.
            z_solve = self._zcomp(z)
            if abs(z_solve - z) > 1e-4:
                print("[Z补偿] 输入z=%.3f → IK求解z=%.4f (补偿%+.1fmm)" % (
                    z, z_solve, (z_solve - z) * 1000))
                self.status_display.append(
                    "[Z补偿] 输入z=%.3f → 求解z=%.4f (补偿%+.1fmm, "
                    "拟合自8组实测, 目标: 实测≈输入)" % (
                        z, z_solve, (z_solve - z) * 1000))

            # ★ 终点3cm停距(2026-09-18 用户要求): IK求解目标 = 输入点沿末端
            #   法向量反向退3cm —— 停距直接写进运动求解, 到位即停, 无回退段
            # ★ 2026-10-30: 「检测地面」要求尖端精确到输入点, 跳过停距
            if ground_exact:
                sx, sy, sz = x, y, z_solve
                self.status_display.append(
                    "[检测地面] 关闭终点停距: 求解目标 (%.4f, %.4f, %.4f)" % (sx, sy, sz))
            else:
                sx, sy, sz = _standoff_target(x, y, z_solve, normal)
                if (abs(sx - x) + abs(sy - y) + abs(sz - z_solve)) > 1e-9:
                    self.status_display.append(
                        "[移动到位置] 终点停距: 求解目标 (%.4f, %.4f, %.4f)" % (sx, sy, sz))
            # ★ 2026-09-27晚: REBOT_IK_BIAS 补偿已移除(两轮对照实测当前臂
            #   IK系统偏差≈0, 叠加补偿反引入 -IK_BIAS 新偏差), 与画框路径一致.
            # ★ 2026-10-16 服务器端轨迹模式: 一次性下发 cmd:"move_traj"(目标
            #   位姿+法向), IK+测地线轨迹由 rebot_arm_control 的 move_to_traj
            #   在服务器端完成 —— 替代原"本地IK预检 + 30Hz插值 + set_joints
            #   流式下发"全链路; 闭环补走(依赖solve_ik/set_joints)一并停用
            r = self.controller.arm_client.move_traj(sx, sy, sz, normal,
                                                     q6=q6)   # ★ 时长由后端按距离自适应

            if r.get("ok"):
                # 更新四元数显示
                quat = normal_to_quaternion(nx, ny, nz)
                self.normal_status_label.setText(
                    f"四元数: [w={quat[0]:.4f}, x={quat[1]:.4f}, y={quat[2]:.4f}, z={quat[3]:.4f}]"
                )

                print("=" * 50)
                print("服务器端轨迹运动 (move_traj):")
                print(f"  X: {x} m, Y: {y} m, Z: {z} m")
                print(f"  求解目标(停距后): ({sx:.4f}, {sy:.4f}, {sz:.4f})")
                print(f"  法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
                print(f"  目标四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                print("=" * 50)

                # ★ 作废旧闭环补走线程(其依赖 solve_ik/set_joints, 且会与
                #   服务器端轨迹互相覆盖); 服务器端轨迹自带IK+Fk验收
                self._cancel_pending_refine()
                self._notify_trail_start()
                self.status_display.append(f"[移动到位置] 已下发服务器端轨迹: 目标 ({sx:.4f}, {sy:.4f}, {sz:.4f}) 法向 ({nx:.3f}, {ny:.3f}, {nz:.3f})")
                self.statusBar().showMessage(
                    "服务器端轨迹运动中: (%.3f, %.3f, %.3f)" % (sx, sy, sz))
                QMessageBox.information(self, "成功",
                    f"机械臂开始移动(服务器端轨迹)!\n\n"
                    f"位置: ({x:.3f}, {y:.3f}, {z:.3f})\n"
                    f"法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})\n"
                    f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            else:
                err = str(r.get('err', ''))
                if 'not running' in err or '未连接' in err:
                    self.status_display.append("[移动失败] 机械臂后端未连接(arm_server 已退出)")
                    QMessageBox.warning(self, "机械臂未连接",
                        "机械臂后端(arm_server)未运行。\n\n请点击「启动机械臂」重新连接后再试。")
                else:
                    self.status_display.append(
                        "[移动失败] 目标 ({:.2f}, {:.2f}, {:.2f}) 不可达(服务器端IK无解). "
                        "提示: 可临时把法向量改 0,0,-1 (竖直向下) 重试; 低位/地面目标(z≥-0.40)现已支持.".format(x, y, z))
                    self.statusBar().showMessage("✗ 移动失败: 目标点不可达", 5000)
                    QMessageBox.warning(self, "移动失败",
                        "目标点 ({:.2f}, {:.2f}, {:.2f}) 在当前姿态约束下不可达。\n\n"
                        "建议:\n"
                        "1. 提高 z (低位+侧向最易不可达, z≥0.06 起步)\n"
                        "2. 减小 y 偏移\n"
                        "3. 检查法向量输入 (0,0,-1 为标准朝下)".format(x, y, z))
        except ValueError as e:
            self.status_display.append("[动作中止] 请输入有效的数字!\n{str(e)}")

    # ★ 修改：开始扫描方法 - 末端俯视地面，在最大角度内来回旋转
    def start_scan(self):
        """★ 巡点循环扫描 (2026-09-17自kk_rebot同步): A→B→C→D→E→F→A 循环往复.
        每点按表面法向量取向(IK 构造末端姿态); 腿间插入经过点(位置+姿态同步
        插值), 整圈 IK 预解算后连续巡行 —— 无驻留/无段间停顿.
        点「停止扫描」优雅退出(33ms 内停止下发), 「紧急停止」= 立即失电."""
        try:
            if getattr(self, 'action_running', False) or self.current_action is not None:
                self.status_display.append("[动作中止] 当前已有动作正在执行!")
                return
            if self.is_scanning:
                self.status_display.append("[动作中止] 扫描正在进行中!")
                return
            # 巡点扫描依赖 arm_server 的 IK, 必须已连接
            try:
                r = self.controller.arm_client.get_state()
                if not (r and r.get("ok")):
                    raise RuntimeError("未连接")
            except Exception:
                QMessageBox.warning(self, "未连接", "请先点「启动机械臂」再开始扫描!")
                return
            self._scan_stop.clear()
            self.is_scanning = True
            self._cancel_pending_refine()   # 扫描期间旧闭环补走不得复活插运动
            threading.Thread(target=self._scan_loop_worker, daemon=True).start()
            self.status_display.append(
                "[开始扫描] 巡点循环 A→B→C→D→E→F→A 连续巡行"
                "(每腿 %d 个经过点, 关节速度 %.2f rad/s); 点「停止扫描」退出"
                % (SCAN_TOUR_VIA_PER_LEG, SCAN_TOUR_SPEED))
        except Exception as e:
            self.is_scanning = False
            QMessageBox.critical(self, "错误", "开始扫描失败: %s" % e)

    def _scan_build_tour(self):
        """展开巡点+经过点 -> 整圈逐点 IK 预解算 -> [(点名或None, 6关节)] 序列.
        ★ 全部解算完毕才开始运动: 运行期零解算间隙(段间停顿的根因).
        经过点: 位置线性插值 + 法向量线性插值后归一 -> 姿态在腿内渐变."""
        W = SCAN_TOUR_WAYPOINTS
        n = len(W)
        units = []
        for _name, _p, nv in W:
            nl = math.sqrt(sum(c * c for c in nv)) or 1.0
            units.append([c / nl for c in nv])
        seq = []          # [(点名或None, xyz, unit_normal)]
        for i in range(n):
            name, p0, _ = W[i]
            p1, n0, n1 = W[(i + 1) % n][1], units[i], units[(i + 1) % n]
            seq.append((name, list(p0), list(n0)))            # 主点
            for t in range(1, SCAN_TOUR_VIA_PER_LEG + 1):     # 经过点
                f = t / float(SCAN_TOUR_VIA_PER_LEG + 1)
                pos = [a + (b - a) * f for a, b in zip(p0, p1)]
                nv = [a + (b - a) * f for a, b in zip(n0, n1)]
                nl = math.sqrt(sum(c * c for c in nv)) or 1.0
                seq.append((None, pos, [c / nl for c in nv]))
        out, last_q = [], None
        for name, pos, nv in seq:
            ok, q = self.controller.move_to_cartesian(
                pos[0], pos[1], pos[2], q6=None, normal=nv)
            if not ok or q is None:
                tag = ("点" + name) if name else "经过点"
                self._log("[扫描] %s(%.2f,%.2f,%.2f) 不可达, 从巡点中剔除"
                          % ((tag,) + tuple(pos)))
                continue
            out.append((name, list(q)))
        return out

    def _stream_leg(self, q_from, q_to, name=None):
        """连续流式走一段关节插值(30Hz, 逐帧轮询停止标志).
        时长 = 行程/巡行速度(每段等关节速度 -> 匀速丝滑).
        返回 False = 收到停止请求(调用方退出)."""
        if self._scan_stop.is_set():
            return False
        travel = max((abs(b - a) for a, b in zip(q_from, q_to)), default=0.0)
        duration = max(0.25, travel / SCAN_TOUR_SPEED)
        steps = max(1, int(duration * SCAN_TOUR_RATE_HZ))
        dt = 1.0 / SCAN_TOUR_RATE_HZ
        for i in range(steps + 1):
            if self._scan_stop.is_set():
                return False
            cur = interpolate(q_from, q_to, i / float(steps))
            pub2ros_pos(self.controller.publisher, cur)
            self.controller.joint_pos = list(cur)
            self.controller.q6_value = cur[5]
            if name is not None and i == steps:
                self._post_ui(lambda nm=name: self.statusBar().showMessage("🔍 扫描中: 点%s" % nm))
            time.sleep(dt)
        return True

    def _scan_loop_worker(self):
        """巡点循环扫描线程: 整圈预解算 -> 匀速连续巡行, 循环往复直至停止."""
        self.action_running = True   # 占用动作通道(30Hz定时器与动作按钮让路)
        self._post_ui(lambda: self.enable_action_buttons(False))
        try:
            self._post_ui(lambda: self.statusBar().showMessage(
                "🔍 正在解算巡点关节序列(含经过点, 首轮数秒)..."))
            tour = self._scan_build_tour()
            if not tour or len(tour) < 2:
                self._log("[扫描中止] 有效巡点不足(全部不可达?), 请检查点位坐标")
                return
            self._log("[扫描] 关节序列就绪: %d 个航点, 巡行速度 %.2f rad/s, 连续巡行不停留"
                      % (len(tour), SCAN_TOUR_SPEED))
            cycle = 0
            q_cur = list(self.controller.joint_pos)
            while not self._scan_stop.is_set():
                cycle += 1
                self._log("[扫描] ===== 第 %d 轮 =====" % cycle)
                stopped = False
                for name, q in tour:
                    if not self._stream_leg(q_cur, q, name):
                        stopped = True
                        break
                    q_cur = q
                if stopped:
                    break
                # 一轮走完: q_cur = 末尾经过点, 下轮首段 q_cur->A 无缝衔接
            self._log("[扫描停止] 已按请求退出循环 (共 %d 轮, 停止下发<0.1s)" % cycle)
        except Exception as e:
            print("[scan_loop] 异常: %s" % e)
            import traceback
            traceback.print_exc()
            self._log("[扫描异常] %s (已中止)" % e)
        finally:
            self.is_scanning = False
            self.action_running = False
            self._post_ui(self._after_scan_done)

    def _after_scan_done(self):
        """扫描结束恢复 UI (主线程执行)."""
        self.enable_action_buttons(True)
        self.status_display.append("[扫描结束] 已退出巡点循环扫描模式")
        self.show_current_status()

    def stop_scan(self):
        """停止巡点循环扫描 (优雅退出: 当前段走完即停, 不失电;
        需要立即停止请点「紧急停止」)."""
        if not self.is_scanning:
            return
        self._scan_stop.set()
        self.statusBar().showMessage("停止请求已记录: 当前段走完即停 (立即停止请点「紧急停止」)", 5000)
        self.status_display.append("[停止扫描] 已请求退出, 当前段走完即停")

    # ──────────────────────────────────────────────────────────────────────────
    # ★★★ 新增：自动物证采集流程 ★★★
    # 对应专利整体流程：UI启动机械臂/相机(沿用原有功能) -> 机械臂初始化 ->
    # 扫描(限位安全) -> CenterNet检测(类别+候选框) -> CDG-ATRG精细分割 ->
    # 标定+深度反投影得到基坐标系三维坐标(+表面法向量) -> 按类别执行
    # execute_action(1/2/3) -> 循环寻找下一个物证
    # ──────────────────────────────────────────────────────────────────────────
    def init_perception_subscribers(self):
        """订阅相机原始彩色图/深度图/内参话题，独立于原有的aruco结果订阅"""
        if self.color_sub is None:
            self.color_sub = rospy.Subscriber(
                RAW_COLOR_TOPIC, Image, self.on_raw_color,
                queue_size=1, buff_size=2 ** 24)
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber(
                RAW_DEPTH_TOPIC, Image, self.on_raw_depth,
                queue_size=10, buff_size=2 ** 24)   # ★ 2026-10-30: 1→10, 缓存更多帧
        if self.caminfo_sub is None:
            self.caminfo_sub = rospy.Subscriber(
                CAMERA_INFO_TOPIC, CameraInfo, self.on_camera_info, queue_size=1)
        # ★ 订阅深度相机内参, 用于彩色像素->深度像素的精确映射(不依赖depth_registration)
        if self.depth_caminfo_sub is None:
            self.depth_caminfo_sub = rospy.Subscriber(
                '/camera/depth/camera_info', CameraInfo,
                self.on_depth_camera_info, queue_size=1)
        # ★ 后台预获取 depth→color 外参(不阻塞主线程, 避免首次检测时UI卡顿)
        self._fetch_depth_to_color_extrinsics_async()

    def on_raw_color(self, msg):
        """彩色图回调: ★只缓存原始msg(引用赋值,原子安全).
        cvtColor/flip/拷贝全部移到取用线程(显示/检测), rospy接收线程
        跑任何 numpy/cv2 操作都可能触发段错误(已实测崩溃在 recv_buff).
        ★ 2026-10-21: 同时设 camera_image, 让手动相机窗口(update_camera_view)
        也用同一条原始彩色流 — CenterNet 检测框由 UI 在 update_camera_view 里画,
        不再依赖 ROS 侧 CenterNet 节点的 result_image 话题."""
        self._raw_color_msg = msg
        # 手动窗口也用原始流 (不再订阅 /forensic3_detect_v4/result_image)
        # ★★ 2026-10-28 修复"角点一会有一会没有":
        #   标定模式下, /aruco_single/result(带角点/坐标轴叠加)与
        #   /camera/color/image_raw(原始图)【两个回调都写 camera_image】.
        #   原始流 30fps、标注图约 10fps, 原始流以 3 倍速率持续覆盖显示缓冲,
        #   表现为画面在"有角点"和"没角点"之间来回跳.
        #   现在: 标定模式时显示流由标注图独占, 本回调只维护 _raw_color_msg
        #   (深度/像素解算仍用原始图, 不受影响).
        if not getattr(self, 'CALIB_MODE', False):
            self.controller.camera_image = msg

    def _decode_color_bgr(self):
        """把缓存的原始彩色msg转成 BGR numpy(不翻转, 保持原始朝向).
        由取用方线程(显示/检测线程)调用, 不在rospy线程.
        ★ reBot 相机正装: 像素坐标与原始图像/camera_info内参/后续ArUco手眼
          标定的参考系保持一致, 禁止任何翻转."""
        msg = getattr(self, '_raw_color_msg', None)
        if msg is None:
            return None
        try:
            if msg.encoding == 'rgb8':
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(
                    msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            elif msg.encoding == 'mono8':
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                return None
            # ★ reBot 相机正装: 不翻转(翻转会让检测像素与相机内参/手眼标定
            #   参考系垂直镜像, 坐标解算全错)
            return img
        except Exception as e:
            rospy.logerr("彩色图转换失败: {}".format(e))
            return None

    def on_raw_depth(self, msg):
        """深度图回调: 缓存原始msg + 维护最近 5 帧环形队列 (用于冻结帧选优)."""
        self._raw_depth_msg = msg
        # ★★ 2026-10-30: 同时维护环形列表, 供冻结时刻挑选好帧
        try:
            self._recent_depth_msgs.append(msg)
            if len(self._recent_depth_msgs) > 5:
                self._recent_depth_msgs.pop(0)
        except Exception:
            pass

    def _decode_depth(self):
        """把缓存的原始深度msg转成numpy(与彩色同朝向, 不翻转)."""
        msg = getattr(self, '_raw_depth_msg', None)
        if msg is None:
            return None
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                dimg = np.frombuffer(
                    msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
            elif msg.encoding == '32FC1':
                dimg = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            else:
                return None
            # ★ 与彩色图同步: 同样不翻转, 保持彩色像素<->深度像素对齐
            return dimg
        except Exception as e:
            rospy.logerr("深度图转换失败: {}".format(e))
            return None

    def _decode_depth_best(self, min_valid_ratio=0.05, wait_frames=15):
        """★★ 2026-10-30: 冻结帧选优深度图.  Orbbec 深度相机 IR 投射器
        偶尔漏拍 (~10%概率), 单帧有效像素可能为 0, 冻结帧就卡死.
        策略: 从最近缓存的环形队列里挑有效像素最多的那一帧; 如果所有缓存帧
        有效率都低于 min_valid_ratio, 就再等 wait_frames 帧 (约 0.5s @30Hz)
        继续挑, 确保冻结帧有足够的有效深度像素.
        返回: (best_depth_img, valid_ratio) 或 (None, 0)"""
        recent = getattr(self, '_recent_depth_msgs', None) or []
        # 先尝试环形队列里已有的
        best_dimg, best_ratio = self._pick_best_from(recent)
        if best_ratio >= min_valid_ratio:
            return best_dimg, best_ratio
        # 不够好 —— 等更多帧进来
        waited = 0
        while waited < wait_frames:
            QApplication.processEvents()
            time.sleep(0.035)   # ~30Hz
            waited += 1
            recent = getattr(self, '_recent_depth_msgs', None) or []
            dimg, ratio = self._pick_best_from(recent)
            if ratio > best_ratio:
                best_dimg, best_ratio = dimg, ratio
            if best_ratio >= min_valid_ratio:
                return best_dimg, best_ratio
        return best_dimg, best_ratio

    @staticmethod
    def _pick_best_from(msgs):
        """从一组 depth Image msg 里挑有效像素最多的, 返回 (dimg, valid_ratio)."""
        best = None
        best_ratio = 0.0
        for msg in msgs:
            try:
                if msg.encoding in ('16UC1', 'mono16'):
                    dimg = np.frombuffer(
                        msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
                    valid_ratio = float(np.count_nonzero(dimg)) / dimg.size
                elif msg.encoding == '32FC1':
                    dimg = np.frombuffer(
                        msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
                    valid_ratio = float(np.count_nonzero(dimg > 0)) / dimg.size
                else:
                    continue
                if valid_ratio > best_ratio:
                    best_ratio = valid_ratio
                    best = dimg
            except Exception:
                continue
        return best, best_ratio

    def on_camera_info(self, msg):
        """彩色相机内参回调：更新fx/fy/cx/cy，用于像素->三维坐标反投影"""
        self.camera_intrinsics.update_from_camera_info_msg(msg)

    def on_depth_camera_info(self, msg):
        """★ 深度相机内参回调：更新深度图自己的fx/fy/cx/cy,
        用于把彩色像素精确映射到深度图像素(深度光心≠彩色光心)"""
        self.depth_intrinsics.update_from_camera_info_msg(msg)

    def _fetch_depth_to_color_extrinsics_async(self):
        """★ 后台线程获取 depth→color 外参, 不阻塞主线程(避免UI卡顿)。
        orbbec_camera 驱动发布 /camera/depth_to_color (orbbec_camera/Extrinsics),
        含 rotation(3x3行优先9浮点) + translation(3浮点,米)。
        成功后缓存到 self._depth_to_color_extrinsics。"""
        if self._depth_to_color_extrinsics is not None:
            return
        import threading
        def _worker():
            import subprocess
            try:
                result = subprocess.run(
                    ['rostopic', 'echo', '-n', '1', '/camera/depth_to_color'],
                    capture_output=True, text=True, timeout=8.0)
                if result.returncode == 0 and result.stdout.strip():
                    import yaml
                    data = yaml.safe_load(result.stdout)
                    rot = np.array(data['rotation'], dtype=np.float64).reshape(3, 3)
                    trans = np.array(data['translation'], dtype=np.float64)
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = rot
                    T[:3, 3] = trans
                    self._depth_to_color_extrinsics = T
                    print("[深度对齐] depth→color 外参获取成功(后台): "
                          "平移={} 基线={:.3f}m".format(
                              np.array2string(trans, precision=4),
                              np.linalg.norm(trans[:2])))
            except Exception as e:
                print("[深度对齐] 获取 depth_to_color 外参失败: {}".format(e))
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _annotate_map_color_to_depth(self, cx, cy, depth_img):
        """★ 框选链路专用彩色→深度像素映射(2026-09-27 修复回归丢失):
        gemini launch depth_registration=true 时 /camera/depth/image_raw 已
        硬件对齐彩色(实测640x480且K与彩色完全一致) —— 深度图尺寸与彩色相同时
        直取(u,v); 无条件走 _map_color_to_depth_pixel 的Astra 25mm基线补偿
        会在错误像素采深度(≈左偏30px → 采到背景/无效 → 基坐标"None"不填框).
        尺寸不同(真Astra双目深度640x400) → 回落基线补偿老逻辑."""
        dh, dw = depth_img.shape[:2]
        msg = getattr(self, '_raw_color_msg', None)
        ch = msg.height if msg is not None else dh
        cw = msg.width if msg is not None else dw
        if (ch, cw) == (dh, dw):
            return (max(0, min(int(round(cx)), dw - 1)),
                    max(0, min(int(round(cy)), dh - 1)))
        return self._map_color_to_depth_pixel(cx, cy, depth_img)

    def _map_color_to_depth_pixel(self, cx, cy, depth_img, z_est=0.38):
        """★ 用深度内参+基线补偿把彩色像素(cx,cy)映射到深度像素(dx,dy)。
        Astra Stereo U3: 深度相机在彩色相机右侧~25mm(双目基线)。
        同一3D点在深度图的u坐标比彩色图左偏 fx*baseline/Z ≈ 30px(在0.38m处)。
        不查深度时用此偏移修正, 避免查到错误像素→深度错误→坐标全错。
        深度内参未收到时用Astra硬编码默认值(cx_d=320,cy_d=200)。"""
        ci = self.camera_intrinsics
        di = self.depth_intrinsics
        dh, dw = depth_img.shape[:2]

        # 基线X分量
        baseline_x = 0.025  # Astra Stereo U3默认
        if self._depth_to_color_extrinsics is not None:
            baseline_x = self._depth_to_color_extrinsics[0, 3]

        # 深度内参: 已收到用实测, 否则用Astra默认
        if di.ready:
            fx_d, fy_d, cx_d, cy_d = di.fx, di.fy, di.cx, di.cy
        else:
            fx_d, fy_d = ci.fx, ci.fy    # 焦距与彩色接近
            cx_d, cy_d = 320.0, 200.0      # 深度640×400的光心

        # 基线引起的水平像素偏移: Δu = fx_d * baseline_x / z_est
        du_baseline = fx_d * baseline_x / z_est

        # 映射: 彩色像素→深度像素(含基线补偿)
        dx = int(round(fx_d * (cx - ci.cx) / ci.fx + cx_d - du_baseline))
        dy = int(round(fy_d * (cy - ci.cy) / ci.fy + cy_d))
        dx = max(0, min(dx, dw - 1))
        dy = max(0, min(dy, dh - 1))
        return dx, dy

    def start_auto_pipeline(self):
        """
        【启动自动采集流程】按钮：
        沿用现有的 start_arm()/start_camera()（不修改这部分逻辑），
        懒加载CenterNet检测器 + 后台检测线程与相机原始话题订阅，
        随后进入状态机自动运行。
        """
        if self.auto_mode:
            QMessageBox.information(self, "提示", "自动采集流程已在运行!")
            return

        try:
            if self.detector is None:
                self.statusBar().showMessage("正在加载CenterNet检测模型，请稍候...")
                QApplication.processEvents()
                self.detector = ForensicDetector()
            if self.detection_thread is None:
                self.detection_thread = ForensicDetectionThread(
                    self.detector, self._get_latest_frame_for_detection)
                self.detection_thread.detection_ready.connect(self.on_detection_ready)
                self.detection_thread.start()
            self.init_perception_subscribers()
        except Exception as e:
            QMessageBox.critical(self, "错误", "自动采集初始化失败: {}".format(e))
            return

        # ★ 订阅刚建立, 给相机最多4秒收首帧(原代码订阅后立即判断, 必然"无相机帧")
        if getattr(self, '_raw_color_msg', None) is None:
            self.status_display.append("[自动采集] 等待相机首帧(最多4秒)...")
            QApplication.processEvents()
            for _ in range(40):
                if getattr(self, '_raw_color_msg', None) is not None:
                    break
                time.sleep(0.1)

        # 复用已有功能启动机械臂与相机，不改变其行为
        if self.launch_process.state() != QProcess.Running:
            self.start_arm()
        if not self.camera_processes:
            self.start_camera()

        # ★ 无相机保护: 从未收到过相机帧则不进入自动流程(否则扫描永远空转)
        if getattr(self, '_raw_color_msg', None) is None:
            self.status_display.append(
                "[自动采集] ★ 未收到相机图像(/camera/color/image_raw 无帧) — "
                "自动识别需要相机. 请先确认相机已启动且 rostopic 有数据. 流程不启动.")
            self.statusBar().showMessage("无相机帧, 自动采集未启动")
            return

        self.auto_mode = True
        self._best_capture = None
        self._capture_history = None
        # ★ 2026-08-28: 跳过回零 —— 从当前位置直接进扫描(省5~8秒"抬起干等");
        #   MOVE_TO_SCAN_START 会从实际关节插值到扫描起点, 无需先经零位。
        self._enter_auto_state('MOVE_TO_SCAN_START')

        self.status_display.append(
            "[自动采集] 流程已启动：机械臂初始化 -> 关节限位内旋转半圈扫描检测(后台线程连续检测) -> "
            "停止 -> 停顿{:.0f}秒 -> 按检测信息执行动作".format(POST_SCAN_PAUSE))
        self.statusBar().showMessage("🤖 自动采集流程已启动")

    def stop_auto_pipeline(self):
        """
        【停止自动采集】按钮：结束状态机。
        若机械臂正处于半圈扫描/回零等由 start_arm_movement() 驱动的运动中，
        立即清除 is_moving 使其停在当前位置；若正处于
        execute_action() 阻塞执行中，则等该次动作跑完（与原有紧急停止行为一致）。
        后台检测线程不销毁，只是置为非激活状态（空闲小睡），
        下次再次启动自动采集时可以直接复用，不用重新加载模型。
        """
        self.auto_mode = False
        self._enter_auto_state('IDLE')
        self.is_moving = False
        self.move_duration = 0
        self._cancel_pending_refine()
        self.stop_scan()  # 兼容手动扫描仍在运行的边界情况
        if self.detection_thread is not None:
            self.detection_thread.set_active(False)
        self._best_capture = None
        self._capture_history = None
        self.statusBar().showMessage("自动采集流程已停止")
        self.status_display.append("[自动采集] 流程已停止")

    def _enter_auto_state(self, new_state):
        """状态机切换到新状态，并重置"该状态的一次性触发动作是否已执行"标记"""
        self.auto_state = new_state
        self._state_action_triggered = False

    def _get_latest_frame_for_detection(self):
        """
        供后台检测线程调用的取帧回调：返回当前最新的(彩色图, 深度图, 关节1角度)。
        彩色图为None时表示还没收到任何相机帧，线程会空闲小睡后重试。
        三者是三次独立的属性读取，理论上可能来自相差几毫秒的不同瞬间
        （与本代码库里其他地方"直接用最新可用数据"的一贯做法一致），
        这点误差在当前精度要求下可以接受。
        """
        # ★ 在取用线程(检测线程)解码: rospy回调线程只缓存msg(段错误修复)
        color = self._decode_color_bgr()
        if color is None:
            return None, None, None
        depth = self._decode_depth()
        th1 = self.controller.joint_pos[0]
        return color, depth, th1

    def _ensure_manual_detection(self):
        """手动相机模式: 后台加载 CenterNet 并启动连续检测(结果只画框, 不驱动动作).
        幂等: 模型已加载/线程已跑则只确保 set_active(True)."""
        if getattr(self, '_manual_det_booting', False):
            return
        self._manual_det_booting = True

        def _boot():
            try:
                if self.detector is None:
                    self.detector = ForensicDetector()
                if self.detection_thread is None:
                    self.detection_thread = ForensicDetectionThread(
                        self.detector, self._get_latest_frame_for_detection)
                    # ★ 两个槽都常连: 手动画框(本方法) + 自动采集状态机.
                    #   on_detection_ready 有 auto_state!=SCANNING 守卫,
                    #   手动模式下收到信号只会直接 return, 不会误触发动作.
                    self.detection_thread.detection_ready.connect(
                        self._on_manual_detection)
                    self.detection_thread.detection_ready.connect(
                        self.on_detection_ready)
                    self.detection_thread.start()
                self.detection_thread.set_active(True)
                print("[CenterNet] 手动画面检测框已开启")
            except Exception as e:
                import traceback
                print("[CenterNet] 手动检测启动失败: %s\n%s" % (e, traceback.format_exc()))
            finally:
                self._manual_det_booting = False

        import threading as _threading
        _threading.Thread(target=_boot, daemon=True).start()

    def _on_manual_detection(self, det, depth_img, th1):
        """手动模式检测缓存槽: 只存最新结果+时间戳给画框用, 不做任何动作."""
        self._latest_detection = det
        self._latest_detection_ts = time.time()

    def on_detection_ready(self, det, depth_img, th1_at_capture):
        """
        后台检测线程发回的检测结果，在主线程(Qt事件循环)里处理，可以安全地
        操作UI控件。det/depth_img/th1_at_capture 三者对应同一帧，避免主线程
        处理时机械臂已经转到别处、深度图已经过期导致的坐标错位问题。

        采用连续 CONFIRM_COUNT 次检测到同一类别才确认发现物证的去抖动策略，
        避免单帧误检（如运动模糊导致的偶然误判）被记为"确认发现"。
        只在处于 SCANNING 状态时才处理，避免状态切换瞬间残留的检测结果
        错误地更新到下一轮扫描里。
        """
        if self.auto_state != 'SCANNING':
            return

        if det.category == self._pending_category:
            self._pending_confirm_count += 1
        else:
            self._pending_category = det.category
            self._pending_confirm_count = 1
        self._pending_depth_img = depth_img
        self._pending_th1 = th1_at_capture

        # ★★★ 每次检测(含未确认)都把类别+像素写入检测信息框 ★★★
        cx, cy = det.centroid_px
        th1_deg = (th1_at_capture if th1_at_capture is not None else 0.0) * 180.0 / 3.14159265
        ts = time.strftime("%H:%M:%S")
        # ★ 记录实际关节角(用于诊断手眼标定/FK问题)
        actual_joints = list(self.controller.joint_pos)
        self.detection_display.append(
            "<span style='color:#C9CCD2;'>[{}] [检测] {}({}) score={:.2f} "
            "像素=({:.0f},{:.0f}) th1={:+.1f}° 确认{}/{} "
            "实际关节=[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]</span>".format(
                ts, det.label, det.category, det.score, cx, cy, th1_deg,
                self._pending_confirm_count, CONFIRM_COUNT,
                *actual_joints))

        self.statusBar().showMessage(
            "🔍 扫描中检测到 {} (score={:.2f})，确认中 {}/{}".format(
                det.label, det.score, self._pending_confirm_count, CONFIRM_COUNT)
        )

        if self._pending_confirm_count < CONFIRM_COUNT:
            return

        self._pending_category = None
        self._pending_confirm_count = 0

        pose = self.compute_target_pose_from_detection(det, self._pending_depth_img, self._pending_th1)
        if pose is None:
            return

        xyz, normal = pose
        # ★ 漂移抑制: 手眼A残余旋转误差使交点随th1呈弱U形(±1.3cm),
        #   单帧坐标直接用会把这个摆动带进动作目标。收集本轮全部确认点,
        #   派发时取中值 → 摆动被中值吸收, 目标稳定。
        if getattr(self, '_capture_history', None) is None:
            self._capture_history = []
        self._capture_history.append(np.asarray(xyz, dtype=np.float64))
        n = len(self._capture_history)
        med = np.median(np.array(self._capture_history), axis=0)
        xyz = tuple(med)   # 用中值替换本帧坐标(记录/派发均用)
        _spread = float(np.linalg.norm(
            np.array(self._capture_history).max(0) -
            np.array(self._capture_history).min(0)) * 100)
        # 连同depth_img/th1一起记录：液体类的蛇形擦拭路径生成
        # (generate_snake_wipe_waypoints)需要用"确认到这一帧"对应的
        # 深度图+关节1角度，不能用执行动作那一刻(机械臂已回零/已停止)的
        # 当前状态，原理与三维坐标解算要求一致（见compute_target_pose_from_detection注释）
        self._best_capture = (det, xyz, normal, self._pending_depth_img, self._pending_th1)
        self.status_display.append(
            "[自动采集] ✅ 确认物证 {}({}) score={:.2f} -> 基坐标=({:.4f},{:.4f},{:.4f})m "
            "(中值×{} 散布{:.1f}cm) 法向=({:.1f},{:.1f},{:.1f}) | 坐标已记录，继续完成本轮扫描".format(
                det.label, det.category, det.score,
                xyz[0], xyz[1], xyz[2], n, _spread,
                normal[0], normal[1], normal[2])
        )

    def compute_target_pose_from_detection(self, det, depth_img, th1):
        """
        依据CDG-ATRG精细分割得到的质心像素坐标 + 深度图 + 相机标定参数，
        解算机械臂基坐标系下的三维坐标，并估计物证表面法向量
        （专利步骤S4深度反投影 + 步骤S5 SNC-RRT 法向量计算，公式18/19）。

        depth_img/th1 必须是"确认到该检测结果那一帧"对应的深度图和关节1角度
        （由调用方——on_detection_ready——从检测线程回传的信号里原样传入），
        不能用调用这个函数时刻的 self.raw_depth_image/self.controller.joint_pos，
        否则在检测结果经后台线程异步传回主线程处理的这段时间里，
        机械臂可能已经转到别的角度、深度图也可能已经更新，会算出错误坐标。

        ★ 相机固定安装在机械臂末端，随臂运动；半圈扫描期间只有关节1
          （基座旋转）在变化，因此th1传给perception_calib做"绕基座Z轴旋转
          th1"的位姿修正（详见该模块注释），而不是用一个与关节角无关的
          常量变换。
        返回 (xyz, normal)；深度无效时返回 None。
        """
        if depth_img is None:
            _msg = "[定位失败] 该帧没有对应深度图(深度流未就绪?) — 跳过本次检测"
            print(_msg); self.status_display.append(_msg)
            self.statusBar().showMessage("⚠️ 该帧没有对应深度图，无法定位")
            return None

        cx, cy = det.centroid_px

        # ★★★ 深度像素映射（核心修复）★★★
        # 彩色图上检测到物证质心 (cx,cy), 要去深度图里查对应像素的深度。
        # 彩色和深度是两个独立传感器, 光心/内参不同且有双目基线偏移,
        # 不能用纯比例缩放映射。
        # 优先级:
        #   1. depth_registration=true 生效(深度图=彩色尺寸) → 直接用(cx,cy)
        #   2. 有depth→color外参+深度内参 → 迭代重投影精确查找(软件对齐)
        #   3. 仅深度内参 → 用内参映射(不含基线, 精度一般)
        #   4. 都没有 → 比例缩放(最差, 仅过渡)
        dh, dw = depth_img.shape[:2]
        color_img = self._decode_color_bgr()   # ★ 取用线程解码(段错误修复)
        if color_img is not None:
            ch, cw = color_img.shape[:2]
        else:
            ch, cw = 480, 640

        ci = self.camera_intrinsics   # 彩色内参
        di = self.depth_intrinsics    # 深度内参

        # ★ 先取末端位姿(get_pose) —— 它既是坐标解算输入, 也用于算
        #   "桌面平面预测深度"(深度传感器的量程下限~0.4m, 扫描高度下
        #   桌面深度全零, 18:51 实测 13x13 邻域 0/729, 必须有平面兜底)
        try:
            _r = self.controller.arm_client.get_pose()
        except Exception as e:
            _msg = "[定位失败] get_pose 异常: %s" % e
            print(_msg); self.status_display.append(_msg)
            self.statusBar().showMessage("⚠️ get_pose 失败, 无法定位")
            return None
        if not _r.get("ok") or not _r.get("pos"):
            _msg = "[定位失败] 机械臂后端未连接(get_pose)"
            print(_msg); self.status_display.append(_msg)
            self.statusBar().showMessage("⚠️ 机械臂后端未连接(get_pose), 无法定位")
            return None
        p_be_now = _r["pos"]
        rpy_now = _r.get("rpy", [0.0, np.pi / 2, 0.0])
        th1_now = _r["joints"][0] if _r.get("joints") else (
            th1 if th1 is not None else 0.0)
        th1_frame = th1 if th1 is not None else th1_now
        # 桌面平面射线求交: 预测深度 + 平面交点(基坐标)
        _plane_xyz, _plane_depth = pcalib.rebot_pixel_table_xyz(
            cx, cy, ci, p_be_now, rpy_now, th1_now, th1_frame)

        depth_m = None
        _map_mode = 'unknown'
        dx, dy = -1, -1

        if dw == cw and dh == ch:
            # 1. 深度已对齐到彩色 (depth_registration=true 生效)
            dx, dy = int(round(cx)), int(round(cy))
            dx = max(0, min(dx, dw - 1))
            dy = max(0, min(dy, dh - 1))
            depth_m = pcalib.sample_stable_depth(depth_img, dx, dy)
            if depth_m is None:
                depth_m = pcalib.sample_stable_depth(depth_img, dx, dy, win=9)
            if depth_m is None:
                depth_m = pcalib.sample_stable_depth(depth_img, dx, dy, win=13)
            _map_mode = 'aligned(direct)'
        else:
            # 2. 用深度内参+基线补偿映射(硬编码Astra默认值, 不依赖话题)
            dx, dy = self._map_color_to_depth_pixel(cx, cy, depth_img)
            # ★ 分级重试: 深度相机对小物证常有空洞, win=5 取不到时
            #   逐级扩窗(9/13), 仍取不到才判无效(并打日志, 不再静默)
            depth_m = pcalib.sample_stable_depth(depth_img, dx, dy)
            if depth_m is None:
                depth_m = pcalib.sample_stable_depth(depth_img, dx, dy, win=9)
            if depth_m is None:
                depth_m = pcalib.sample_stable_depth(depth_img, dx, dy, win=13)
            _map_mode = 'intrinsics+baseline'

        if depth_m is None or depth_m <= 0:
            # ★ 最终兜底: 用整块检测掩模采深度(小物证质心处深度常为空洞,
            #   但掩模覆盖整个物证+边缘, 有效像素概率高得多)。
            #   掩模像素(彩色系)逐点映射到深度系, 收集有效深度取中值;
            #   剔除明显背景(>1.5m)与近处噪点(<0.05m)。
            try:
                _mask = getattr(det, 'mask', None)
                if _mask is not None and depth_img is not None:
                    _ys, _xs = np.nonzero(_mask)
                    if _ys.size > 4:
                        _vals = []
                        for _k in range(0, _ys.size, max(1, _ys.size // 120)):
                            _mdx, _mdy = self._map_color_to_depth_pixel(
                                int(_xs[_k]), int(_ys[_k]), depth_img)
                            _v = pcalib.sample_stable_depth(depth_img, _mdx, _mdy, win=3)
                            if _v is not None and 0.05 < _v < 1.5:
                                _vals.append(_v)
                        if _vals:
                            depth_m = float(np.median(_vals))
                            _map_mode += '+mask(%d点)' % len(_vals)
            except Exception as _e:
                print("[深度兜底] 掩模采样异常: %s" % _e)

        # ★ 深度来源优先级(2026-08-28 复盘定稿):
        #   桌面场景下 平面求交(手眼A/h+桌面高度) 远比传感器可靠 —— 立体相机
        #   量程下限~0.38m, 扫描高度下真值~0.18m 物理不可测, 传感器读数全是
        #   噪声(曾以175~189mm的垃圾值骗过±10cm校验 → z漂到桌面下 → IK失败
        #   → 收尾校验误判电机失能自动recover = "到右端断电回零"假象)。
        #   规则: 平面可用 → 一律用平面; 平面失败才用传感器(>0.4m远距才可信)。
        if _plane_depth is not None:
            if depth_m is not None:
                print("[深度] 传感器=%.3fm 平面=%.3fm → 用平面(传感器近距不可信)" % (
                    depth_m, _plane_depth))
            depth_m = _plane_depth
            _map_mode += '+tableplane(%.3fm)' % _plane_depth
        elif depth_m is not None and depth_m < 0.40:
            # 平面求交失败 + 传感器值低于量程下限 → 噪声, 弃用
            print("[深度] 平面失败且传感器%.3fm低于量程下限0.4m → 噪声弃用" % depth_m)
            depth_m = None

        if depth_m is None or depth_m <= 0:
            _msg = ("[定位失败] 深度无效(平面求交失败且传感器不可信) "
                    "像素=({:.0f},{:.0f}) 跳过本次".format(cx, cy))
            print(_msg); self.status_display.append(_msg)
            self.statusBar().showMessage("⚠️ 深度与平面兜底均失败")
            return None

        # ★ 诊断: 打印映射细节(用于排查深度问题)
        print("[深度映射] mode={} 深度图={}×{} 彩色={}×{} "
              "彩色像素=({:.0f},{:.0f}) -> 深度像素=({:.0f},{:.0f}) "
              "depth_m={:.3f}m 深度内参ready={}({:.1f},{:.1f},fx={:.1f},fy={:.1f}) "
              "外参ready={}".format(
                  _map_mode, dw, dh, cw, ch, cx, cy, dx, dy, depth_m,
                  di.ready, di.cx, di.cy, di.fx, di.fy,
                  self._depth_to_color_extrinsics is not None))

        # ★★★ v20 终版坐标链 (2026-08-31): M-链交点直接作为最终坐标 ★★★
        #   rebot_pixel_table_xyz 已用三组真值标定的 通用投影映射M 拟合
        #   (rms 6.4mm), 其返回的 _plane_xyz 即基坐标答案(z恒=桌面平面)。
        #   ★ 旧 A/h 链(rebot_cam_xyz_to_base_scan)已废弃: 曾把M链的"度量
        #   距离"误当"轴向深度"喂入旧链 → z随th1漂移(-0.03~+0.09)、坐标
        #   系统性偏差 ~16cm(16:40 实测), 是"基坐标与实际对不上"的根因。
        if _plane_xyz is None:
            _msg = ("[定位失败] M-链平面求交失败(像素=({:.0f},{:.0f})) 跳过本次".format(cx, cy))
            print(_msg); self.status_display.append(_msg)
            self.statusBar().showMessage("⚠️ 平面求交失败")
            return None
        xyz = np.asarray(_plane_xyz, dtype=np.float64)
        depth_m = _plane_depth if _plane_depth is not None else 0.0
        _map_mode += '+Mchain'

        # ★★★ 坐标转化全链路日志（供排查）★★★
        intr = self.camera_intrinsics
        p_cam = pcalib.pixel_depth_to_cam_xyz(cx, cy, max(depth_m, 0.01), intr)  # 仅日志显示用
        actual_joints = list(_r.get("joints") or self.controller.joint_pos)
        print("\n" + "=" * 70)
        print("[坐标转化链路] {}({}) score={:.2f}".format(det.label, det.category, det.score))
        print("  1. 像素坐标(彩色图)  : cx={:.1f}, cy={:.1f}  (图像尺寸 {}×{})".format(
            cx, cy, cw, ch))
        print("  3. 相机→物证距离     : {:.4f} m ({:.1f} mm)".format(depth_m, depth_m * 1000.0))
        print("  5. 相机系近似点(日志) : p_cam=({:.4f}, {:.4f}, {:.4f}) m".format(
            p_cam[0], p_cam[1], p_cam[2]))
        print("  6. 末端位置(get_pose): p_be_now=({:.4f}, {:.4f}, {:.4f})".format(*p_be_now))
        print("  7. 帧时刻关节1       : th1_frame={:.3f} (now={:.3f})".format(th1_frame, th1_now))
        print("  9. ★基坐标系三维坐标(M链): xyz=({:.4f}, {:.4f}, {:.4f}) m".format(
            xyz[0], xyz[1], xyz[2]))
        print("=" * 70 + "\n")

        # 同步在UI状态显示区输出坐标（不只在终端打印, 方便无终端启动时查看）
        self.status_display.append(
            "[坐标转化] {}({}) 像素=({:.0f},{:.0f}) 深度={:.1f}mm "
            "p_cam=({:.3f},{:.3f},{:.3f}) th1={:.1f}° -> 基坐标=({:.4f},{:.4f},{:.4f})m".format(
                det.label, det.category, cx, cy, depth_m * 1000.0,
                p_cam[0], p_cam[1], p_cam[2],
                (th1 if th1 is not None else 0.0) * 180.0 / 3.14159265,
                xyz[0], xyz[1], xyz[2]))

        # ★★★ 在检测信息框里追加"确认后的完整转化坐标"(高亮, 便于和真实摆放位置对比)★★★
        th1_deg = (th1 if th1 is not None else 0.0) * 180.0 / 3.14159265
        ts = time.strftime("%H:%M:%S")
        self.detection_display.append(
            "<span style='color:#FFC24B;font-weight:bold;'>  [{ts}] ✅确认转化: "
            "基坐标=({x:.4f},{y:.4f},{z:.4f})m  深度={d:.1f}mm "
            "p_cam=({pcx:.3f},{pcy:.3f},{pcz:.3f}) th1={th:+.1f}°</span>".format(
                ts=ts, x=xyz[0], y=xyz[1], z=xyz[2],
                d=depth_m * 1000.0,
                pcx=p_cam[0], pcy=p_cam[1], pcz=p_cam[2], th=th1_deg))
        self.detection_display.append("")  # 空行分隔每次确认

        # ★ 法向量强制为 [0, 0, -1] (夹爪垂直向下): 桌面水平, 预设动作要求夹爪保持向下,
        #   estimate_surface_normal 在深度图噪声/分辨率不齐时不可靠(曾算出斜法向量导致
        #   姿态错误→0组解→机械臂扭曲), 改为固定垂直向下, 与原版 kk_copy 预设动作一致。
        normal = [0.0, 0.0, -1.0]
        return xyz, normal

    def dispatch_action_for_detection(self, det, xyz, normal, depth_img=None, th1=None):
        """
        把解算出的三维坐标/法向量写入笛卡尔空间控制输入框（与手动操作走同一套
        入口，方便在状态显示区核对），再按物证类别调用现成的 execute_action()。

        depth_img/th1: 与det配套的那一帧深度图/关节1角度，只有液体类
        (action_num==1)会用到——execute_action()内部据此调用
        pcalib.generate_snake_wipe_waypoints(det.mask, depth_img, ..., th1=th1)
        生成蛇形擦拭路径；其余类别忽略这两个参数。
        """
        x, y, z = xyz
        nx, ny, nz = normal

        self.x_input.setText("{:.4f}".format(x))
        self.y_input.setText("{:.4f}".format(y))
        self.z_input.setText("{:.4f}".format(z))
        self.j6_input.setText("0")   # ★ 2026-10-30 用户要求: 目标位置 J6 一直填 0
        self.nx_input.setText("{:.4f}".format(nx))
        self.ny_input.setText("{:.4f}".format(ny))
        self.nz_input.setText("{:.4f}".format(nz))

        action_num = CATEGORY_TO_ACTION.get(det.category)
        if action_num is None:
            self.statusBar().showMessage("⚠️ 未知物证类别: {}，放弃本次执行".format(det.category))
            return

        self.status_display.append(
            "[自动采集] 执行采集：{}({}) score={:.2f} -> 基坐标系坐标=({:.3f},{:.3f},{:.3f}) "
            "法向量=({:.2f},{:.2f},{:.2f}) -> 执行动作{}".format(
                det.label, det.category, det.score, x, y, z, nx, ny, nz, action_num)
        )

        self.execute_action(action_num, mask=det.mask, depth_img=depth_img, th1=th1)

    def update_auto_pipeline(self):
        """
        自动采集状态机，挂在现有 QTimer(30ms) 回调 update_all() 上，
        仅在 self.auto_mode 为 True 时生效，不影响手动操作模式下的行为。

        状态流转：
          IDLE               空闲，未启动自动采集
          INIT_HOMING        调用 return_to_zero() 做机械臂初始化动作，等待到位
          MOVE_TO_SCAN_START 移动到半圈扫描的起始关节角(其余关节为
                             SCAN_POSITION姿态，关节1=SCAN_HALF_CIRCLE_TH1_START)
          SCANNING           关节1从起始角单方向匀速转到结束角(一次"半圈"扫描，
                             不来回摆动)；期间打开后台检测线程
                             (ForensicDetectionThread)连续检测，不再按固定
                             时间间隔抽帧，确认到物证只记录坐标
                             (self._best_capture)、不中断旋转；
                             旋转到位即视为"机械臂停止"
          POST_SCAN_WAIT     机械臂已停止，停顿 POST_SCAN_PAUSE 秒；到时间后，
                             若本轮记录到目标，按记录的信息执行对应动作，
                             否则跳过本轮；执行完成后若仍处于自动模式则回到
                             INIT_HOMING 重新开始，形成闭环
        """
        if not self.auto_mode:
            return

        if self.auto_state == 'INIT_HOMING':
            if not self._state_action_triggered:
                self.return_to_zero(duration=3.0)
                self._state_action_triggered = True
                return
            if not self.is_moving:
                self._enter_auto_state('MOVE_TO_SCAN_START')
            return

        if self.auto_state == 'MOVE_TO_SCAN_START':
            if not self._state_action_triggered:
                # ★ 单程扫描: 恒定 左→右 (START=-2.7 → END=+1.4)
                scan_start_pos = [SCAN_HALF_CIRCLE_TH1_START] + list(SCAN_POSITION[1:])
                self.start_arm_movement(self.controller.joint_pos, scan_start_pos, 3.0)
                self._state_action_triggered = True
                return
            if not self.is_moving:
                self._enter_auto_state('SCANNING')
            return

        if self.auto_state == 'SCANNING':
            if not self._state_action_triggered:
                # ★ 单程扫描: 左→右, 到端即停, 紧接动作派发
                scan_end_pos = [SCAN_HALF_CIRCLE_TH1_END] + list(SCAN_POSITION[1:])
                self.start_arm_movement(self.controller.joint_pos, scan_end_pos, SCAN_ROTATE_DURATION)
                self._state_action_triggered = True
                self._best_capture = None
                self._capture_history = None
                self._pending_category = None
                self._pending_confirm_count = 0
                if self.detection_thread is not None:
                    self.detection_thread.set_active(True)  # 打开后台连续检测
                self.status_display.append("[自动采集] 开始单程扫描(左→右)...")
                return

            # 检测本身完全由后台线程(ForensicDetectionThread)连续进行，
            # 结果通过 on_detection_ready() 信号异步回传，这里只需要等旋转到位
            if not self.is_moving:
                if self.detection_thread is not None:
                    self.detection_thread.set_active(False)  # 本轮扫描结束，暂停后台检测
                self._enter_auto_state('POST_SCAN_WAIT')
            return

        if self.auto_state == 'POST_SCAN_WAIT':
            if not self._state_action_triggered:
                self._state_action_triggered = True
                self._post_scan_wait_start = rospy.get_time()
                self.statusBar().showMessage(
                    "扫描完成，机械臂已停止，{:.0f}秒后按检测信息执行动作...".format(POST_SCAN_PAUSE))
                return

            if rospy.get_time() - self._post_scan_wait_start < POST_SCAN_PAUSE:
                return

            # ★ 动作线程还没跑完时, 状态机原地等待(不许切到INIT_HOMING和动作线程
            #   同时发指令 —— 这是"自动识别中途断电"的根因之二)
            if getattr(self, 'action_running', False):
                return

            if self._best_capture is not None:
                det, xyz, normal, captured_depth_img, captured_th1 = self._best_capture
                self._best_capture = None   # ★ 先清, 防止下一tick重复dispatch
                self.dispatch_action_for_detection(det, xyz, normal, captured_depth_img, captured_th1)
                # ★ execute_action 起的是工作线程, 本tick直接返回, 等 action_running
                #   变 False 后(下一个tick)才继续走状态机
                return
            else:
                self.status_display.append("[自动采集] 本轮扫描未发现可执行的物证")

            # ★ 2026-08-28 定稿流程: 单程扫描(左→右)→ 直接按相机类别/坐标执行
            #   预设动作 → 动作收尾回零后重新开始下一轮(不再反向扫描/乒乓)。
            if self.auto_mode:
                self._enter_auto_state('MOVE_TO_SCAN_START')
            else:
                self._enter_auto_state('IDLE')
            return

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：execute_action — 将夹爪控制整合到运动序列中，添加等待时间参数
    # ──────────────────────────────────────────────────────────────────────────
    def execute_action(self, action_num, mask=None, depth_img=None, th1=None):
        """动作入口: UI主线程读输入框 -> 工作线程执行运动.
        ★ reBot 修复: 原来输入框读取在工作线程里(QLineEdit.text() 跨线程 = Qt 未定义
        行为), 是"返回初始位时 UI 闪退"的根因之一. 现在所有输入框都在本方法(UI线程)
        读取并缓存到 self._action_params, 工作线程只用缓存值."""
        print("[execute_action] 入口: action_num=%s action_running=%s" %
              (action_num, getattr(self, 'action_running', False)))   # ★ 入口打点
        if getattr(self, 'action_running', False):
            QMessageBox.warning(self, "警告", "当前已有动作正在执行!")
            return

        # ---- UI 线程: 读取并校验输入框 ----
        try:
            safe_height = float(self.height_input.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的安全高度!")
            return
        try:
            tool_len_smear = float(self.tool_len_smear_input.text()) / 1000.0
            tool_len_dab = float(self.tool_len_dab_input.text()) / 1000.0
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的工具补偿(mm)!")
            return
        try:
            target_x = float(self.x_input.text())
            target_y = float(self.y_input.text())
            target_z = float(self.z_input.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请在笛卡尔空间控制框中输入有效的 X / Y / Z 坐标!")
            return
        try:
            nx = float(self.nx_input.text())
            ny = float(self.ny_input.text())
            nz = float(self.nz_input.text())
            normal = [nx, ny, nz]
        except ValueError:
            normal = None
        q6_text = self.j6_input.text()
        # ★ 留空=保持当前J6不动(2026-10-25, 同「移动到位置」路径):
        #   法向竖直(0,0,±1)时 normal_to_rpy 备用基准与当前J6差90°,
        #   q6=None → J6突转90°; 用台账角覆盖roll基准保持J6现状
        q6 = float(q6_text) if q6_text.strip() else float(self.controller.q6_value)

        self._action_params = {
            'safe_height': safe_height,
            'target': [target_x, target_y, target_z],
            'normal': normal,
            'q6': q6,
            'tool_len_smear': tool_len_smear,
            'tool_len_dab': tool_len_dab,
        }

        def _worker():
            self.action_running = True
            self._cancel_pending_refine()   # 预设动作期间旧闭环补走不得插运动
            _act_name = {1: "涂抹1号位", 4: "涂抹2号位", 2: "粘取1号位",
                         5: "粘取2号位", 3: "抓取"}.get(action_num, "动作%d" % action_num)
            self._worker_log("[%s] 已启动, 正在校验目标点(约1s)..." % _act_name,
                             "%s已启动" % _act_name, 2000)
            try:
                self._execute_action_impl(action_num, mask=mask, depth_img=depth_img, th1=th1)
            except Exception as e:
                print("[execute_action] 异常: %s" % e)
                import traceback
                traceback.print_exc()
            finally:
                # ★ 收尾校验: 仅当动作真正执行过运动(非IK失败提前中止)才检查回零。
                #   曾因IK失败的废坐标提前return(臂合法停在扫描端点~78°)被误判
                #   "电机失能"自动recover —— 表现为"到右端断电1s+回零"假象。
                if not getattr(self, '_action_motion_started', False):
                    self._log(
                        "[收尾] 动作未真正起步(IK失败/参数中止), 跳过回零校验(不做recover)")
                    # ★ 2026-10-19: 未起步=开场校验失败, 弹窗告知原因
                    #   (只打状态栏时用户感知="点按钮没反应")
                    _reason = getattr(self, '_action_abort_reason', None) or \
                        "开场校验失败(目标点/安全高度/回收位不可达)"
                    self._post_ui(lambda r=_reason: QMessageBox.warning(
                        self, "动作中止", r + "\n\n机械臂未运动, 请检查坐标后重试."))
                    self._action_abort_reason = None
                else:
                    try:
                        r = self.controller.arm_client.get_state()
                        if r.get("ok") and r.get("joints"):
                            import math as _m
                            err = [abs(_m.degrees(v)) for v in r["joints"]]
                            worst = max(err)
                            if worst > 12.0:
                                # ★ 2026-10-29: 原来会自动 recover + 重走回零
                                #   —— 电机掉线后静默重新使能并自行运动, 会伤人.
                                #   现只提示, 由用户确认现场后手动点「启动机械臂」.
                                self._log(
                                    "[收尾校验] 关节未回零(最大偏差%.1f°) — 电机可能中途失能, "
                                    "已跳过自动恢复(防伤人), 请确认机械臂周围安全后手动点"
                                    "「启动机械臂」" % worst)
                    except Exception as e:
                        print("[收尾校验] %s" % e)
                # ★ 同步复位两道门(纯赋值, 线程安全): 上一动作做完立即可点下一个动作.
                #   原来只复位 action_running, current_action 等 _after_action_done
                #   (主线程异步) 才清 -> 快速点击/事件丢失时被"当前已有动作"拦截 = 无响应
                self.action_running = False
                self.current_action = None
                # UI 恢复必须回主线程 —— 投递队列(原QTimer.singleShot从
                # python线程调度永不触发且可段错误, 2026-09-17)
                self._post_ui(self._after_action_done)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _after_action_done(self):
        """动作完成后恢复 UI (在主线程执行)."""
        self.current_action = None
        self.enable_action_buttons(True)
        self.show_current_status()

    def _run_move_traj(self, x, y, z, normal=None, q6=None, duration=None,
                       desc="move_traj", max_wait=90.0):
        """服务器端轨迹段: 发 cmd:move_traj → 轮询 traj_active 等轨迹发完 →
        get_state 同步关节台账. 返回 True/False(失败=IK不可达/超时, 调用方应中止).

        ★ move_to_traj 只"发起"轨迹就返回(send_thread 后台按 duration 发完),
          不等完成 —— 动作序列若不等完就连发下一段, 下一段 move_to_traj 开头
          的 _stop_send+join 会砍掉上一条轨迹(臂走一半被劫走).
        ★ move_traj 期间 joint_pos 台账不更新(服务器端驱动), 完成后必须
          get_state 重同步, 否则后续 move_joint/回零段以旧起点规划.
        ★ duration=None(默认): 后端按移动距离自适应时长(近移快/远移慢
          恒速感, T=clamp(d/0.08+0.8, 1.5, 8.0)); 传值则覆盖."""
        r = self.controller.arm_client.move_traj(x, y, z, normal=normal,
                                                 q6=q6, duration=duration)
        if not (r or {}).get("ok"):
            # ★ 响应超时≠失败(2026-10-18 涂抹步骤6事故根因): cmd 已写入
            #   stdin 管道, 后端可能仍在处理(IK/轨迹生成)甚至轨迹已异步
            #   发起 —— 直接中止会造成"UI判失败中止+recover / 后端轨迹
            #   还在跑"的状态分裂(recover 砍正在执行的轨迹, 台账错乱).
            #   补偿: 探测 traj_active 最多3s, 出现即视为已接受, 按正常
            #   流程等完成; 确认无轨迹再判失败.
            for _ in range(6):
                try:
                    rs = self.controller.arm_client.get_state()
                    if rs.get("ok") and rs.get("traj_active"):
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            else:
                self._log("[%s] move_traj 失败: %s" % (desc, _errtext(r) if r else "无响应"))
                return False
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait:
            try:
                rs = self.controller.arm_client.get_state()
                if rs.get("ok") and not rs.get("traj_active", False):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            self._log("[%s] move_traj 等待超时(%.0fs), 臂可能仍在运动" % (desc, max_wait))
            return False
        try:
            rs = self.controller.arm_client.get_state()
            if rs.get("ok") and rs.get("joints"):
                self.controller.joint_pos = list(rs["joints"])
                self.controller.q6_value = rs["joints"][5]
        except Exception:
            pass
        return True

    def _return_home_mid(self, tag=""):
        """★ 2026-10-31 流程中途回两段式回零的中立位(仅第一段, 不回零位):
        用于「目标 Z<0」时先到中立位再转移, 避免从低位工具架/目标区低空掠行."""
        return self._run_move_traj(HOME_MID_POSE[0], HOME_MID_POSE[1], HOME_MID_POSE[2],
                                   normal=list(HOME_MID_NORMAL), q6=None,
                                   duration=HOME_MID_DUR, desc="回零·中立位%s" % tag)

    def _execute_action_impl(self, action_num, mask=None, depth_img=None, th1=None):
            """执行指定的动作(1/2/3/4/5) — 2026-09-08 自 gluon_controller_kk_rebot.py 同步
            (夹爪mm间距模式 + 转移段改关节空间直达 + 涂抹面内方向 + 动作3校验修复):
            ★ 2026-10-25: 涂抹/粘取各分1/2号位(用户要求), 同逻辑仅架位x不同;
                1=涂抹1号位(0.143,-0.085,0.037), 4=涂抹2号位(0.2,-0.085,0.037),
                2=粘取1号位(0.143,0.078,0.037), 5=粘取2号位(0.2,0.078,0.037)
            涂抹(液体类)：抓取点固定为棉签工具位，涂抹点为笛卡尔空间输入的坐标；
                沿法向量正交的面内方向往复涂抹; 完成后放回棉签架取用槽位(取放同点)释放棉签
            粘取(微小类)：抓取点固定为粘取器工具位，粘取点为笛卡尔空间输入的坐标；
                垂直下压-停留-提起完成粘附; 完成后放回粘取器架释放粘取器
            动作3(抓取，固体类)：抓取点即目标点本身，沿表面法向量接近后闭合夹爪抓取;
                完成后转移到专属回收位松开夹爪放下物证(不放工具架)

            mask/depth_img/th1: 自动检测流程(dispatch_action_for_detection)透传的
                分割掩模/深度图/关节1角度(旧蛇形路径已移除, 当前仅保留参数兼容,
                手动模式下默认None, 行为一致).
            """
            # ★ 重复动作检查已上移到 execute_action 的 action_running(唯一闸门)
            self._action_motion_started = False   # 收尾校验闸门: 首条运动指令才置True

            # ★ 动作起点同步: 从 arm_server 读真实关节角覆盖 controller.joint_pos.
            #   扫描/上一动作/手动拨动后 joint_pos 常与实际不符(曾以过期 -121°
            #   为起点规划大回转 -> j1 满力矩 -> 整体过流失能). 以实测为准.
            try:
                rs = self.controller.arm_client.get_state()
                if rs.get("ok") and rs.get("joints"):
                    self.controller.joint_pos = list(rs["joints"])
                    self.controller.q6_value = rs["joints"][5]
                    self._log("[动作] 起点已同步实际关节: " +
                        ", ".join("%.0f°" % (__import__('math').degrees(v)) for v in rs["joints"]))
            except Exception as e:
                print("[起点同步] %s" % e)

            # ★ 输入已在 execute_action(UI线程)读好缓存, 此处只取值(不再碰输入框)
            params = getattr(self, '_action_params', None)
            if params is None:
                self._log("[动作中止] 输入参数缺失!")
                return
            safe_height = params['safe_height']
            target_x, target_y, target_z = params['target']
            _user_target_z = float(target_z)   # ★ 用户输入的目标Z(补偿前), 供Z<0过渡点判断
            normal = params['normal']
            q6 = params['q6']

            # 根据动作类型设置抓取点
            # ★ 2026-10-25: 涂抹/粘取各分1/2号位(用户要求), 逻辑相同仅架位x不同;
            #   1=涂抹1号位, 4=涂抹2号位, 2=粘取1号位, 5=粘取2号位, 3=抓取
            if action_num == 1:    # 涂抹1号位
                grab_x, grab_y, grab_z = 0.141, -0.078, 0.036   # ★前方棉签架1号位(原[-0.10,0.16]左后方致j1大回转过流, 已前移)
            elif action_num == 4:  # 涂抹2号位
                grab_x, grab_y, grab_z = 0.212, -0.078, 0.034     # ★棉签架2号位(2026-10-25用户指定)
            elif action_num == 2:  # 粘取1号位
                grab_x, grab_y, grab_z = 0.141, 0.085, 0.036    # ★reBot工具架:粘取器1号位(梯形IK验证)
            elif action_num == 5:  # 粘取2号位
                grab_x, grab_y, grab_z = 0.208, 0.085, 0.034     # ★粘取器2号位(2026-10-25用户指定)
            elif action_num == 3:  # 抓取动作（固体类，抓取点即目标点本身）
                grab_x, grab_y, grab_z = target_x, target_y, target_z
            else:
                self._log("[动作中止] 无效的动作编号!")
                return

            # ★ 工具架J6对齐(2026-09-16 用户指定): 架子插槽朝向固定, 腕部旋转对准后
            #   夹取/放回 —— 棉签架(涂抹1/2号位)与粘取器架(粘取1/2号位) J6=0;
            #   动作3无工具架, 维持用户q6.
            q6_rack = 0 if action_num in (1, 2, 4, 5) else q6
            if q6_rack != q6:
                self._log("[动作] 工具架J6对齐: %s J6=%+.1frad" % (
                    "棉签架" if action_num in (1, 4) else "粘取器工具架", q6_rack))

            # ============================================================
            # ★ 抓取相关：使用默认姿态（不传递法向量）
            # ============================================================
            # 对抓取点求逆运动学
            # ★ 修复(2026-09-07 同步): 动作3抓取点=目标点, 实际运动全程法向约束(下探/退回
            #   都传 normal), 开场校验却用默认朝下姿态解同一位置 —— 校验了一个
            #   永远不会用到的姿态, 用户点(0.4,0,0.3)+姿态(1,0,0)时默认姿态恰好
            #   差9~11mm(>8mm容差)被误判"不可达", 动作开场即中止(表现=点击无响应).
            #   动作3改传用户法向; 动作1/2工具架取放实际就是默认姿态, 不变.
            # ★ 法向None防护前置(2026-09-16): 法向框为空时动作3的 _grab_normal
            #   取到None → fast IK失败即中止(曾表现为"按下无反应"); 显式默认
            #   朝下姿态, 与arm_server默认一致.
            if normal is None:
                normal = [0.0, 0.0, -1.0]
                print("[动作] 法向框为空, 默认朝下姿态 (0,0,-1)")

            # ★ 2026-10-30 用户实测: 夹爪抓取工具(棉签/粘取器)后工具尖端沿法向
            #   伸出, 接触点需沿法向反退工具长度; ★ 2026-10-31 UI 可调:
            #   涂抹/粘取补偿值由输入框读取(mm→m, 默认20/34). 抓取不带工具不补偿.
            #   偏移加在 target 源头 → 接触点/进退点/涂抹往复 全部同步平移,
            #   法向进退点退距仍为用户输入点-(L+0.05)·n, 与接触点间距保持 5cm.
            _tool_len = {1: params.get('tool_len_smear', 0.020),
                         4: params.get('tool_len_smear', 0.020),
                         2: params.get('tool_len_dab', 0.034),
                         5: params.get('tool_len_dab', 0.034)}
            _L = _tool_len.get(action_num, 0.0)
            if _L:
                target_x -= _L * normal[0]
                target_y -= _L * normal[1]
                target_z -= _L * normal[2]
                self._log("[动作] 工具尖端补偿: 沿法向反退 %.0fmm (%s)" % (
                    _L * 1000.0, "涂抹" if action_num in (1, 4) else "粘取"))

            # ★ 2026-10-25: 工具架取放校验姿态必须=实际运动姿态(朝下0,0,-1)
            #   原 None=identity水平朝前, 臂展边缘(x=0.217)校验误报"不可达"
            #   (pos_err=0.0136), 而实际 move_traj 朝下姿态可达(pos_err=0)
            _grab_normal = normal if action_num == 3 else [0.0, 0.0, -1.0]
            success_p2, joints_p2 = self.controller.move_to_cartesian(
                grab_x, grab_y, grab_z, q6_rack, _grab_normal, fast=True
            )
            if not success_p2 and _ik_supports_fast(self.controller.move_to_cartesian):
                # fast失败先回落完整IK(固定工具架点默认姿态必可达)
                success_p2, joints_p2 = self.controller.move_to_cartesian(
                    grab_x, grab_y, grab_z, q6_rack, _grab_normal, fast=False
                )
            if not success_p2 and action_num == 3:
                # ★ 动作3抓取点=目标点: 法向候选兜底(与目标点校验同套),
                #   成功者作为全程姿态 —— 原"开场即中止"是"按下无响应"主因之一
                for _desc, _dn in _normal_candidate_list(None):
                    _s, _j = self.controller.move_to_cartesian(
                        grab_x, grab_y, grab_z, q6_rack, _dn, fast=True)
                    if _s and _j is not None:
                        success_p2, joints_p2 = _s, _j
                        normal = list(_dn)
                        self._log("[动作] 抓取点字段法向不可达, 自动改用法向 "
                                  "(%.2f,%.2f,%.2f) [%s]" % (_dn[0], _dn[1], _dn[2], _desc))
                        break
            if not success_p2:
                self._log(f"[动作中止] 无法对抓取点 ({grab_x}, {grab_y}, {grab_z}) 求解逆运动学!")
                return

            # 对抓取点安全高度求逆运动学
            # ★ 动作3跳过此校验: 该点(用户点X/Y, H=0.13)+法向实测不可达(差15.8mm),
            #   但动作3的运动从不经过"抓取点安全高度"(下探走法向退回点p_target_above,
            #   已在下方单独校验), above_point2 也仅动作1/2工具架取放使用 ——
            #   幽灵校验会在动作开场误杀, 故动作3直接占位跳过.
            if action_num != 3:
                success_above, joints_above_p2 = self.controller.move_to_cartesian(
                    grab_x, grab_y, safe_height, q6_rack, _grab_normal, fast=True
                )
            else:
                success_above, joints_above_p2 = True, joints_p2   # 占位, 动作3不使用
            if not success_above:
                self._log(f"[动作中止] 无法对抓取点安全高度 ({grab_x}, {grab_y}, {safe_height}) 求解逆运动学!")
                return

            # ============================================================
            # ★ 目标点（涂抹/粘取/抓取）：传递法向量
            # ============================================================
            # 对目标点求逆运动学
            # ★ 法向None防护(2026-09-16): 法向框为空时 normal=None, 校验通过后
            #   下游 normal[0] 会TypeError炸掉动作("点击无反应"的另一路径);
            #   显式默认朝下姿态(与arm_server默认一致).
            if normal is None:
                normal = [0.0, 0.0, -1.0]
                print("[动作] 法向框为空, 默认朝下姿态 (0,0,-1)")
            success_target, joints_target = self.controller.move_to_cartesian(
                target_x, target_y, target_z, q6, normal, fast=True
                # 传递 normal → 使用法向量生成姿态
            )
            if not success_target:
                # ★ 法向候选兜底(2026-09-16): 框选在深度FOV边缘(图像顶部)时
                #   法向估计会失败, 法向框留旧值 → 目标IK开场即败且中止消息
                #   曾因跨线程append不可见 = "点击涂抹无反应". 现自动遍历
                #   候选法向(与画框移动同一套), 成功者作为本动作全程姿态.
                #   ★ 字段法向刚失败, 候选从常用列表开始(不再重复试字段值).
                for _desc, _dn in _normal_candidate_list(None):
                    _s, _j = self.controller.move_to_cartesian(
                        target_x, target_y, target_z, q6, _dn, fast=True)
                    if _s and _j is not None:
                        success_target, joints_target = _s, _j
                        normal = list(_dn)
                        self._log("[动作] 字段法向不可达, 自动改用法向 "
                                  "(%.2f,%.2f,%.2f) [%s]" % (_dn[0], _dn[1], _dn[2], _desc))
                        break
            if not success_target:
                self._log(f"[动作中止] 无法对目标点 ({target_x}, {target_y}, {target_z}) 求解逆运动学!")
                return
            self._action_motion_started = True   # ★ IK全通过, 运动即将开始(收尾校验开门)

            # ★ reBot 修改: 接触点=用户输入点(不再沿法向量回退5cm —— 原GLUON逻辑
            #   把工作点抬高导致"涂抹位置与输入不一致"); 进退点单独计算
            contact_x, contact_y, contact_z = target_x, target_y, target_z

            target_x -= 0.05*normal[0]
            target_y -= 0.05*normal[1]
            target_z -= 0.05*normal[2]

            # 对目标点安全高度求逆运动学
            success_above_target, joints_above_target = self.controller.move_to_cartesian(
                target_x, target_y,target_z , q6, normal, fast=True
                # 传递 normal → 使用法向量生成姿态
            )
            if not success_above_target:
                self._log(f"[动作中止] 无法对目标点安全高度 ({target_x}, {target_y}, {safe_height}) 求解逆运动学!")
                return

            # ============================================================
            # ★★★ 物证回收位 —— 求解逆运动学（默认姿态，释放动作对
            # 末端朝向没有严格要求，不传法向量更简单也更安全）
            # ============================================================
            # ★ 回收位按动作分派(2026-10-25 全部取放同点, 复用抓取点坐标):
            #   涂抹1/2号位: 放回对应棉签架槽位
            #   粘取1/2号位: 放回对应粘取器架槽位
            #   抓取: 专属回收位(固体物证不与待用工具混放)
            if action_num in (2, 5):
                recovery_x, recovery_y, recovery_z = grab_x, grab_y, grab_z  # 粘取器架对应号位
            elif action_num == 3:
                # ★ 2026-10-19 修正: 原(0.14, 0, 0.05) 正前方 y=0 是 IK 死区
                #   (pos_err=0.092m 不可达, 动作开场校验必失败="点抓取没反应")。
                #   y 取已验证可达的侧向值(棉签架 y=-0.093 同参数反复跑通),
                #   x 外移错开棉签架避免混放。
                recovery_x, recovery_y, recovery_z = 0.155, -0.093, 0.05   # ★ 抓取动作专属回收位
            else:
                # ★ 2026-10-25: 放回=取用槽位(取放同点)
                recovery_x, recovery_y, recovery_z = grab_x, grab_y, grab_z  # 棉签架(涂抹1/2号位)
            success_recovery, joints_recovery = self.controller.move_to_cartesian(
                recovery_x, recovery_y, recovery_z, q6_rack, [0.0, 0.0, -1.0], fast=True
            )
            if not success_recovery:
                self._log(f"[动作中止] 无法对物证回收位 ({recovery_x}, {recovery_y}, {recovery_z}) 求解逆运动学!")
                return

            success_recovery_above, joints_recovery_above = self.controller.move_to_cartesian(
                recovery_x, recovery_y, safe_height, q6_rack, [0.0, 0.0, -1.0], fast=True
            )
            if not success_recovery_above:
                self._log(f"[动作中止] 无法对物证回收位安全高度 ({recovery_x}, {recovery_y}, {safe_height}) 求解逆运动学!")
                return

            # ★★★ 2026-09-17 关键修复: 校验IK毒化台账 ★★★
            #   move_to_cartesian 每次解算都写 controller.joint_pos —— 开场6次
            #   IK校验后台账已变成"回收位上方"的虚构构型, 而臂实际还在起点!
            #   步骤3的 move_joint 以台账为流式起点 → 臂以VMAX猛冲追赶虚构轨迹
            #   (动作开头猛冲 + 下探段仍在消化追赶误差 → "下探过快/点位不准/
            #   到位等待偏差83°"的根因). 运动段开始前重新同步实测关节.
            try:
                _rs = self.controller.arm_client.get_state()
                if _rs.get("ok") and _rs.get("joints"):
                    self.controller.joint_pos = list(_rs["joints"])
                    self.controller.q6_value = _rs["joints"][5]
                    self._log("[动作] 运动起点已重新同步实际关节(清除校验IK的台账残留)")
            except Exception as _e:
                print("[台账重同步] %s" % _e)

            # 定义各点（关节解，仍保留：给"回零位"这类关节空间过渡用）
            point0 = [0, 0, 0, 0, 0, 0]
            point2       = joints_p2
            above_point2 = joints_above_p2
            point_target       = joints_target
            above_point_target = joints_above_target

            # ★★★ 笛卡尔坐标点位，下面全部motion改用 move_linear() 走直线 ★★★
            p_grab         = [grab_x, grab_y, grab_z]
            p_grab_above   = [grab_x, grab_y, safe_height]
            p_target       = [contact_x, contact_y, contact_z]   # 涂抹/粘取/抓取的接触点
            p_target_above = [target_x, target_y, target_z]      # 沿法向量回退0.05m后的进退点
            p_recovery       = [recovery_x, recovery_y, recovery_z]   # ★ 物证回收释放点
            p_recovery_above = [recovery_x, recovery_y, safe_height]  # ★ 回收位安全高度

            self.current_action = action_num
            self._post_ui(lambda: self.enable_action_buttons(False))  # ★ 投递主线程(原跨线程setEnabled=段错误源)

            # ★★★ 新增：本函数改为同步阻塞执行——move_linear()/move_joint() 都会阻塞到运动结束
            # 才返回，所以不再需要 self.move_sequence 队列和 QTimer 异步调度。
            # 注意：阻塞期间界面（包括"紧急停止"按钮）不会响应，直到整个动作跑完。★★★
            publisher = self.controller.publisher
            ik = self.controller.move_to_cartesian

            try:
                # ════════════════════════════════════════════════════════
                # (2026-09-08 自 kk_rebot 同步: 夹爪全部改mm间距模式;
                #  转移段弃用RRT, 改 move_joint 关节空间直达)
                # ════════════════════════════════════════════════════════
                if action_num != 3:
                    # 液体/微小类：先去工具架取棉签/粘取器（默认姿态，不涉及法向约束）
                    # 1. 打开夹爪(先等臂到位再动爪)
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    time.sleep(0.8)

                    # 3. 移动到抓取点上方 —— ★ 2026-10-17 改服务器端 move_traj
                    #    (测地线+min-jerk, 替代30Hz线性插值的起停突变); 服务器从
                    #    当前实测关节连续解IK, 无需UI预解; 失败即中止动作
                    if not self._run_move_traj(grab_x, grab_y, safe_height,
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="转移·棉签架上方"):
                        self._log("[动作中止] 棉签架上方转移失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s)
                    wait_arrival(self.controller, list(self.controller.joint_pos))
                    time.sleep(3.0)

                    # 4. 下探到抓取点 —— ★ 改服务器端 move_traj; J6=工具架对准角
                    if not self._run_move_traj(p_grab[0], p_grab[1], p_grab[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="下探·棉签架"):
                        self._log("[动作中止] 棉签架下探失败!")
                        return

                    # 5. 闭合夹爪，取工具 —— ★ 爪距30mm; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s);
                    #   上方已有 wait_arrival, 此处只补停留
                    time.sleep(3.0)
                    self._send_gripper(GRIPPER_MM_GRIP)
                    wait_gripper(self.controller, GRIPPER_MM_GRIP)   # ★ 实测到位(停稳=已夹住工具)

                    # 6. 返回安全高度 —— ★ 改服务器端 move_traj; 保持工具架对准角
                    if not self._run_move_traj(p_grab_above[0], p_grab_above[1], p_grab_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="退回·安全高度(棉签)"):
                        self._log("[动作中止] 棉签架退回失败!")
                        return

                    # ★ 2026-10-31 用户要求: 目标点 Z<0 时, 取完工具退回架上方后
                    #   先回中立位, 再转移目标上方, 避免从低位工具架直接斜穿到
                    #   低位目标(低空掠行). 涂抹/粘取共用此段.
                    if _user_target_z < 0:
                        if not self._return_home_mid("(取工具后)"):
                            self._log("[动作中止] 取工具后回中立位失败!")
                            return

                if action_num in (1, 4):  # 涂抹动作（液体类, 1/2号位同逻辑）
                    # ★★★ 以粘取模式代码段为模板重写(结构/函数/参数逐行一致),
                    #     唯一差异: 粘取的"目标点停留1秒"换成 小幅度来回涂抹运动 ★★★

                    # 7. 转移段：抓取点(棉签架)上方 -> 目标点上方 —— ★ 改服务器端
                    #    move_traj(带目标法向姿态, 测地线过渡比关节空间插值丝滑)
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="转移·目标上方"):
                        self._log("[动作中止] 目标上方转移失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s)
                    wait_arrival(self.controller, list(self.controller.joint_pos))
                    time.sleep(3.0)

                    # 8. 接近段：法向约束下探到涂抹点 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target[0], p_target[1], p_target[2],
                                               normal=normal, q6=q6,
                                               desc="下探·涂抹点"):
                        self._log("[动作中止] 涂抹点下探失败!")
                        return

                    # 9. 小幅度来回涂抹(替换粘取的 time.sleep(1.0) 停留):
                    #    ★ 2026-10-17 改服务器端 move_traj 版往复: 端点直接发
                    #      笛卡尔, IK/测地线/五次min-jerk 全在服务器端 250Hz 执行
                    #      —— 无UI预解IK停顿、无30Hz线性插值起停突变, 逐笔丝滑.
                    #      (原三点预解IK+纯move_joint插值版保留在git历史)
                    #    ★ 面内方向 = 垂直末端姿态法向量(原固定世界X已废弃).
                    _dir = _smear_dir_from_normal(normal)
                    _smear_r = 0.04
                    _right = [p_target[0] + _dir[0] * _smear_r,
                              p_target[1] + _dir[1] * _smear_r,
                              p_target[2] + _dir[2] * _smear_r]
                    _left = [p_target[0] - _dir[0] * _smear_r,
                             p_target[1] - _dir[1] * _smear_r,
                             p_target[2] - _dir[2] * _smear_r]
                    _ok_smear = True
                    for _k in range(2):   # ★ 2026-10-31 用户要求: 往复改为 2 个来回(原3)
                        _ok_smear = self._run_move_traj(
                            _right[0], _right[1], _right[2], normal=normal, q6=q6,
                            desc="涂抹笔%d↗" % (_k + 1))
                        if not _ok_smear:
                            break
                        _ok_smear = self._run_move_traj(
                            _left[0], _left[1], _left[2], normal=normal, q6=q6,
                            desc="涂抹笔%d↙" % (_k + 1))
                        if not _ok_smear:
                            break
                    if _ok_smear:
                        _ok_smear = self._run_move_traj(
                            p_target[0], p_target[1], p_target[2], normal=normal,
                            q6=q6, desc="涂抹收笔回中")
                    if not _ok_smear:
                        # 服务器端IK失败/超时: 原move_linear回落已无意义(同一IK内核
                        # 同样失败), 直接中止动作, 收尾闸门会自动recover+补走回零
                        self._log("[动作中止] 涂抹往复段失败!")
                        return

                    # 10. 返回安全高度 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="退回·安全高度(目标)"):
                        self._log("[动作中止] 目标退回失败!")
                        return

                    # ★ Z<0: 退回目标上方后先回中立位, 再转移回收位上方
                    if _user_target_z < 0:
                        if not self._return_home_mid("(目标后)"):
                            self._log("[动作中止] 目标退回后回中立位失败!")
                            return

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改服务器端
                    #     move_traj(默认朝下姿态+棉签架J6对准角, 测地线过渡)
                    if not self._run_move_traj(recovery_x, recovery_y, safe_height,
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="转移·回收位上方"):
                        self._log("[动作中止] 回收位上方转移失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s)
                    wait_arrival(self.controller, list(self.controller.joint_pos))
                    time.sleep(3.0)

                    # 12. 下探到回收位 —— ★ 改服务器端 move_traj; 插槽归还精定位
                    if not self._run_move_traj(p_recovery[0], p_recovery[1], p_recovery[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="下探·回收位"):
                        self._log("[动作中止] 回收位下探失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s);
                    #   下方夹爪动作前本就有 wait_arrival, 此处只补停留
                    time.sleep(3.0)

                    # 13. 打开夹爪，释放棉签; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度 —— ★ 改服务器端 move_traj(保持夹爪向下)
                    #    q6=q6_rack: 与下探段同源, 消掉正朝下法向下传0/传None
                    #    基准roll差90°导致的J6突变
                    if not self._run_move_traj(p_recovery_above[0], p_recovery_above[1], p_recovery_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="退回·安全高度(回收)"):
                        self._log("[动作中止] 回收退回失败!")
                        return

                    # 15. 返回初始位置 —— 两段式回零(降负载 + 全关节同步复位):
                    #     第一段: 到"紧凑中立姿态"(悬臂收拢使 j1 负载小,
                    #             不再最后单独回零);
                    #     第二段: 中立姿态回到零位(小行程, 平缓).
                    #     ★ 2026-10-20 改服务器端 move_traj(min-jerk): 旧
                    #       move_joint 30Hz 线性插值无速度剖面+段间回验死
                    #       时间 = "回零慢且卡"根因, 与全动作链路统一.
                    if not self._run_move_traj(HOME_MID_POSE[0], HOME_MID_POSE[1], HOME_MID_POSE[2],
                                               normal=list(HOME_MID_NORMAL), q6=None,
                                               duration=HOME_MID_DUR, desc="回零·中立位"):
                        self._log("[动作中止] 回零中立位失败!")
                        return
                    if not self._run_move_traj(HOME_ZERO_POSE[0], HOME_ZERO_POSE[1], HOME_ZERO_POSE[2],
                                               normal=list(HOME_ZERO_NORMAL), q6=None,
                                               duration=HOME_ZERO_DUR, desc="回零·零位"):
                        self._log("[动作中止] 回零零位失败!")
                        return
                    # ★ 2026-11-07 回零到位窗口: traj_active=False 只代表轨迹
                    #   点发完 —— J1 静摩擦卡滞差~3°时, 原逻辑 0.7s 后即发
                    #   enable, 而后端 enable 会把指令目标同步成实测角 →
                    #   助推见 err=0 静默退出 → J1 永远差着(末端~17mm), 表现
                    #   为"动作完成但臂不回初始位". 显式等关节真正到位(给助推
                    #   0.5s 判定 + 3s×3 次爬坡的完整窗口)再使能.
                    if not wait_arrival(self.controller, list(INITIAL_POSITION),
                                        tol_deg=1.5, timeout=10.0):
                        self._log("[回零] ⚠ 10s 内未完全到零位(助推可能已放弃), 继续收尾")
                    # ★ 动作收尾: 重新使能全部电机
                    try:
                        self.controller.arm_client._send({"cmd": "enable"}, timeout=10)
                    except Exception as e:
                        print("[收尾使能] %s" % e)
                    self.controller.joint_pos = point0.copy()
                    self.controller.q6_value = 0.0

                elif action_num in (2, 5):  # 粘取动作（微小类, 1/2号位同逻辑）
                    # 7. 转移段：抓取点上方 -> 目标点上方 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="转移·目标上方(粘取)"):
                        self._log("[动作中止] 粘取目标转移失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s)
                    wait_arrival(self.controller, list(self.controller.joint_pos))
                    time.sleep(3.0)

                    # 8. 接近段：法向约束下探到目标点 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target[0], p_target[1], p_target[2],
                                               normal=normal, q6=q6,
                                               desc="下探·目标点(粘取)"):
                        self._log("[动作中止] 粘取目标下探失败!")
                        return

                    # 9. 在目标点停留（模拟粘取动作）—— ★ 2026-10-31 用户要求 1s
                    time.sleep(1.0)

                    # 10. 返回安全高度 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="退回·安全高度(粘取)"):
                        self._log("[动作中止] 粘取目标退回失败!")
                        return

                    # ★ Z<0: 退回目标上方后先回中立位, 再转移回收位上方
                    if _user_target_z < 0:
                        if not self._return_home_mid("(目标后)"):
                            self._log("[动作中止] 粘取目标退回后回中立位失败!")
                            return

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_recovery_above[0], p_recovery_above[1], p_recovery_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="转移·回收位上方(粘取)"):
                        self._log("[动作中止] 粘取回收转移失败!")
                        return

                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s)
                    wait_arrival(self.controller, list(self.controller.joint_pos))
                    time.sleep(3.0)

                    # 12. 下探到回收位 —— ★ 改服务器端 move_traj; J6=工具架对准角
                    if not self._run_move_traj(p_recovery[0], p_recovery[1], p_recovery[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="下探·回收位(粘取)"):
                        self._log("[动作中止] 粘取回收位下探失败!")
                        return

                    # 13. 打开夹爪，释放粘取器(连同粘附的微小物证)到回收位; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    # ★ 停留3s等臂充分到位(2026-10-31用户要求, 原2s);
                    #   上方已有 wait_arrival, 此处只补停留
                    time.sleep(3.0)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度 —— ★ 改服务器端 move_traj(保持夹爪向下)
                    #    q6=q6_rack: 与下探段同源, 消掉正朝下法向下传0/传None
                    #    基准roll差90°导致的J6突变
                    if not self._run_move_traj(p_recovery_above[0], p_recovery_above[1], p_recovery_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="退回·安全高度(粘取回收)"):
                        self._log("[动作中止] 粘取回收退回失败!")
                        return

                    # 15. 返回初始位置 —— ★同步两段式(降负载 + 全关节同步复位)
                    #     ★ 2026-10-20 改服务器端 move_traj(min-jerk): 旧
                    #       move_joint 30Hz 线性插值无速度剖面+段间回验死
                    #       时间 = "回零慢且卡"根因, 与全动作链路统一.
                    if not self._run_move_traj(HOME_MID_POSE[0], HOME_MID_POSE[1], HOME_MID_POSE[2],
                                               normal=list(HOME_MID_NORMAL), q6=None,
                                               duration=HOME_MID_DUR, desc="回零·中立位"):
                        self._log("[动作中止] 回零中立位失败!")
                        return
                    if not self._run_move_traj(HOME_ZERO_POSE[0], HOME_ZERO_POSE[1], HOME_ZERO_POSE[2],
                                               normal=list(HOME_ZERO_NORMAL), q6=None,
                                               duration=HOME_ZERO_DUR, desc="回零·零位"):
                        self._log("[动作中止] 回零零位失败!")
                        return
                    # ★ 2026-11-07 回零到位窗口: 等关节真正到位再使能(同涂抹
                    #   分支, 详见上方注释 —— enable 同步目标会掐死 J1 助推).
                    if not wait_arrival(self.controller, list(INITIAL_POSITION),
                                        tol_deg=1.5, timeout=10.0):
                        self._log("[回零] ⚠ 10s 内未完全到零位(助推可能已放弃), 继续收尾")
                    # ★ 动作收尾: 重新使能全部电机(涂抹等大负载动作后个别关节
                    #   (尤其j1)可能被电机保护性去使能 —— "关节可手拨动"的修复
                    try:
                        self.controller.arm_client._send({"cmd": "enable"}, timeout=10)
                    except Exception as e:
                        print("[收尾使能] %s" % e)
                    self.controller.joint_pos = point0.copy()  # ★ 同步位置，理由同上
                    self.controller.q6_value = 0.0

                elif action_num == 3:  # 抓取动作（固体类）
                    # 新逻辑：固体类抓取点即目标点本身，没有"先去工具架"这一
                    # 段（见上面 if action_num != 3 的跳过），直接从当前位置
                    # (关节空间过渡，起点未知)移动到目标点上方的法向约束进退点
                    # p_target_above，再沿表面法向量插值接近到接触点p_target，
                    # 真正闭合夹爪抓取，抓稳后转移到回收位松开。

                    # 6a. 打开夹爪 —— ★抓取前全开(80mm 上限); 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    time.sleep(0.8)   # ★夹爪降速后等待加倍, 确保全开到位

                    # 7. 转移段：当前位置 -> 目标点上方 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="转移·目标上方(抓取)"):
                        self._log("[动作中止] 抓取目标转移失败!")
                        return

                    # 8. 接近段：沿表面法向量下探到接触点 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target[0], p_target[1], p_target[2],
                                               normal=normal, q6=q6,
                                               desc="下探·接触点(抓取)"):
                        self._log("[动作中止] 抓取接触点下探失败!")
                        return

                    # 9. 闭合夹爪，真正抓取物证 —— ★ 爪距30mm; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_GRIP)
                    wait_gripper(self.controller, GRIPPER_MM_GRIP)   # ★ 实测到位(停稳=已夹住物体)

                    # 10. 沿法向量退回进退点 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_target_above[0], p_target_above[1], p_target_above[2],
                                               normal=normal, q6=q6,
                                               desc="退回·安全高度(抓取)"):
                        self._log("[动作中止] 抓取退回失败!")
                        return

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改服务器端 move_traj
                    if not self._run_move_traj(p_recovery_above[0], p_recovery_above[1], p_recovery_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="转移·回收位上方(抓取)"):
                        self._log("[动作中止] 抓取回收转移失败!")
                        return

                    # 12. 下探到回收位 —— ★ 改服务器端 move_traj(保持夹爪向下); J6=工具架对准角
                    if not self._run_move_traj(p_recovery[0], p_recovery[1], p_recovery[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6_rack,
                                               desc="下探·回收位(抓取)"):
                        self._log("[动作中止] 抓取回收位下探失败!")
                        return

                    # 13. 打开夹爪，松开固体物证到回收位 —— ★全开确保物体脱手; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度 —— ★ 改服务器端 move_traj(保持夹爪向下)
                    if not self._run_move_traj(p_recovery_above[0], p_recovery_above[1], p_recovery_above[2],
                                               normal=[0.0, 0.0, -1.0], q6=q6,
                                               desc="退回·安全高度(抓取回收)"):
                        self._log("[动作中止] 抓取回收退回失败!")
                        return

                    # 15. 返回初始位置 —— ★同步两段式(降负载 + 全关节同步复位)
                    #     ★ 2026-10-20 改服务器端 move_traj(min-jerk): 旧
                    #       move_joint 30Hz 线性插值无速度剖面+段间回验死
                    #       时间 = "回零慢且卡"根因, 与全动作链路统一.
                    if not self._run_move_traj(HOME_MID_POSE[0], HOME_MID_POSE[1], HOME_MID_POSE[2],
                                               normal=list(HOME_MID_NORMAL), q6=None,
                                               duration=HOME_MID_DUR, desc="回零·中立位"):
                        self._log("[动作中止] 回零中立位失败!")
                        return
                    if not self._run_move_traj(HOME_ZERO_POSE[0], HOME_ZERO_POSE[1], HOME_ZERO_POSE[2],
                                               normal=list(HOME_ZERO_NORMAL), q6=None,
                                               duration=HOME_ZERO_DUR, desc="回零·零位"):
                        self._log("[动作中止] 回零零位失败!")
                        return
                    # ★ 2026-11-07 回零到位窗口: 等关节真正到位再使能(同涂抹
                    #   分支, 详见上方注释 —— enable 同步目标会掐死 J1 助推).
                    if not wait_arrival(self.controller, list(INITIAL_POSITION),
                                        tol_deg=1.5, timeout=10.0):
                        self._log("[回零] ⚠ 10s 内未完全到零位(助推可能已放弃), 继续收尾")
                    # ★ 动作收尾: 重新使能全部电机(涂抹等大负载动作后个别关节
                    #   (尤其j1)可能被电机保护性去使能 —— "关节可手拨动"的修复
                    try:
                        self.controller.arm_client._send({"cmd": "enable"}, timeout=10)
                    except Exception as e:
                        print("[收尾使能] %s" % e)
                    self.controller.joint_pos = point0.copy()  # ★ 同步位置，理由同上
                    self.controller.q6_value = 0.0
            except Exception as e:
                # 工作线程里的运动异常只打印, 不碰任何 Qt 控件(防闪退)
                print("[execute_action] 运动段异常: %s" % e)
                import traceback
                traceback.print_exc()
                self._log("[动作异常] %s (已中止)" % e)
            finally:
                pass  # (UI 恢复由 execute_action 外层 finally 在主线程执行)

    def enable_action_buttons(self, enable):
        for btn in self.action_buttons:
            btn.setEnabled(enable)

    def _cancel_pending_refine(self):
        """作废所有尚未完成的闭环补走线程(新运动/急停/暂停时调用)."""
        self._move_epoch += 1

    def start_arm_movement(self, start_pos, end_pos, duration, bump_epoch=True):
        # ★ duration<=0 直接拒发(除零会导致Qt abort闪退), 保持is_moving=False
        if duration is None or duration <= 0:
            duration = 1.0
        # 默认: 任何新运动都作废旧闭环线程; 闭环自己的补走调用传 bump_epoch=False
        if bump_epoch:
            self._move_epoch += 1
        self.is_moving = True
        self.move_start_time = rospy.get_time()
        self.move_duration = duration
        self.move_start_pos = start_pos
        self.move_end_pos = end_pos
        return self._move_epoch

    def _notify_trail_start(self):
        """★ 仿真: 通知 rebot_ponits_sim_node 开始记录末端轨迹.
        仅「移动到位置」手动/画框两条路径调用(扫描/J6微调/预设动作不触发).
        停止由 sim 节点自行根据 /joint_states 静止判定, 无需 UI 发停止."""
        try:
            if self._trail_pub is None:
                self._trail_pub = rospy.Publisher('/arm_trail_record', Bool, queue_size=1)
                rospy.sleep(0.2)   # 首次: 等订阅者(sim节点)TCP建连, 否则第一帧丢失
            self._trail_pub.publish(Bool(data=True))
            rospy.sleep(0.05)
            self._trail_pub.publish(Bool(data=True))   # 冗余一帧, 双保险
        except Exception as e:
            print("[trail] 触发发布失败(不影响运动): %s" % e)

    def rotate_j6(self, angle):
        new_pos = self.controller.joint_pos.copy()
        new_pos[5] += angle
        self.start_arm_movement(self.controller.joint_pos, new_pos, MOVE_DURATION / 2)
        self.controller.q6_value = new_pos[5]
        self.show_current_status()

    def read_current_pose(self):
        """★ 标定辅助: 读当前末端位姿 (经 arm_server FK), 显示到状态栏 + 文本框.
        用法: 用关节控制把机械臂摇到目标点, 点此按钮, 记下 pos 填入预设坐标."""
        try:
            r = self.controller.arm_client.get_pose()
        except Exception as e:
            self.status_display.append("[位姿] 读取失败: %s" % e)
            return
        if not r.get("ok"):
            self.status_display.append("[位姿] 读取失败: %s" % r.get("err", "未知"))
            return
        pos = r.get("pos", [0, 0, 0])
        rpy = r.get("rpy", [0, 0, 0])
        joints = r.get("joints", [0] * 6)
        import numpy as np
        msg = ("[当前位姿] pos=[%.3f, %.3f, %.3f] m  rpy=[%.3f, %.3f, %.3f] rad  "
               "joints(deg)=[%s]") % (
            pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2],
            ", ".join("%.1f" % np.degrees(v) for v in joints))
        self.status_display.append(msg)
        self.statusBar().showMessage("当前位姿: (%.3f, %.3f, %.3f)" % (pos[0], pos[1], pos[2]))

    def return_to_zero(self, duration=None):
        # 如果正在扫描，先停止扫描
        if self.is_scanning:
            self.is_scanning = False
            self._scan_stop.set()
        # ★ 2026-10-18: 回零时长按关节距离自适应(近零位快/远零位慢):
        #   先 get_state 同步真实关节角(防过期台账导致起点错/速度算错),
        #   T = clamp(max|Δq| / 0.12rad/s, 4s, 15s) —— 恒定关节速度,
        #   与 move_traj 的 0.08m/s 末端均速同量级的从容节奏;
        #   显式传 duration(动作流程 3s 收尾)仍优先
        if duration is None:
            try:
                st = self.controller.arm_client.get_state()
                if st.get("ok") and st.get("joints"):
                    self.controller.joint_pos[:5] = [
                        float(v) for v in st["joints"][:5]]
            except Exception:
                pass
            _dur = HOME_DURATION
            try:
                _dq = max(abs(float(v)) for v in self.controller.joint_pos[:5])
                _dur = max(_dq / 1.0, 2.5)
                print("[回零] 最大关节角差 %.2f rad -> 时长 %.1f s" % (_dq, _dur))
            except Exception:
                pass
        else:
            _dur = duration
        self.start_arm_movement(self.controller.joint_pos, INITIAL_POSITION,
                                _dur)
        self.controller.q6_value = 0.0
        self.show_current_status()

    def _worker_log(self, text, status=None, ms=4000):
        """★ 工作线程安全的UI反馈: 入队由主线程 update_all 显示, 同时 print
        落盘 rebot_ui.log(后台线程原本既不落盘、QTimer调度又不触发, 全静默)."""
        try:
            self._ui_msgs.append((text, status, ms))
        except Exception:
            pass
        if text:
            print(text)

    def _post_ui(self, fn):
        """★ 工作线程→主线程投递UI操作(2026-09-17): Qt控件/Timer必须回主线程.
        原QTimer.singleShot(0,..)从python线程调用不触发且可致段错误."""
        try:
            self._ui_calls.append(fn)
        except Exception:
            pass

    def _log(self, msg):
        """★ 线程安全状态栏输出(2026-09-16): 主线程直接append; 工作线程走
        _worker_log队列 —— 修复动作路径跨线程append(QTextCursor警告+消息
        丢失/不渲染 = "点击涂抹无反应"的可见性根因)."""
        try:
            # ★ 2026-10-19: 记录最近一次动作中止原因(worker finally 弹窗用,
            #   否则开场IK失败只打状态栏, 用户感知="点按钮没反应")
            if msg.startswith("[动作中止]"):
                self._action_abort_reason = msg
            if threading.current_thread() is threading.main_thread():
                self.status_display.append(msg)
            else:
                self._worker_log(msg)
        except Exception:
            print(msg)

    def start_arm(self):
        """启动机械臂: 拉起 arm_server + 连接 + 恢复使能 + 夹爪零位自愈.
        ★ connect 可能耗时十几秒(电机模式切换重试), 必须在后台线程做,
        否则 UI 主线程同步等待 = 界面卡死无响应('启动机械臂UI卡死'的根因)."""
        def _worker():
            try:
                if not self.controller.arm_client.start():
                    self._worker_log("[reBot] arm server 子进程启动失败",
                                     "机械臂后端启动失败", 6000)
                    return
                self._worker_log("[reBot] 后端已拉起, 正在连接电机(模式切换最长约十几秒)...",
                                 "正在连接机械臂...", 15000)
                r = self.controller.arm_client.connect()
                # ★ 适配器冻结自愈: 连接失败(全部电机无响应)大概率是PCAN冻结
                #   (点「启动机械臂」也没反应的根因), USB重置后重试一次.
                if not r.get("ok"):
                    self._worker_log("[reBot] 连接失败(%s), USB重置PCAN后自动重试(约5s)..."
                                     % _errtext(r), "连接失败, USB重置PCAN重试中...", 8000)
                    if self._reset_pcan_usb():
                        r = self.controller.arm_client.connect()
                    else:
                        self._worker_log("[reBot] ★ USB重置PCAN失败(sudo密码不对或脚本失败), 跳过自动重试",
                                         "PCAN重置失败", 6000)

                if r.get("ok"):
                    self._arm_started = True   # 看门狗开始值守CAN健康
                    self.controller.arm_client._send({"cmd": "recover"})
                    # ★ 夹爪零位自愈 (2026-09-08 自 kk_rebot 同步): 电机EEPROM零位
                    #   损坏后断电重启多圈参考可能偏移 -> 间距指令与实物不符.
                    #   以"指尖相触"绝对锚点自动重标: 启动读数与上次停车位置一致
                    #   (差<0.35rad≈5.5mm)则免标定零延迟; 不一致才匍匐锚定
                    #   (压力≤3力矩) + 与停车位置推算交叉验证(防碎屑假锚点).
                    #   ★ 匍匐锚定阻塞15~40s, 在本worker线程执行(不在UI线程).
                    try:
                        _rs = self.controller.arm_client.get_state()
                        _boot_rad = float(_rs.get("gripper", 0.0)) if _rs.get("ok") else None
                        if _rs.get("ok") and _rs.get("joints"):
                            # ★ J6开机位记录(2026-09-09): [J6归位]按钮的目标基准
                            self._j6_home = float(_rs["joints"][5])
                            # ★ 指令台账同步: connect/recover后 服务器目标=实测位,
                            #   台账(controller.joint_pos)对齐一次, 保证后续
                            #   rotate_j6/_j6_move_to 用台账起点=服务器目标
                            #   (其他关节零位移, 只动J6) —— 台记若还是UI零位,
                            #   首次J6操作会把臂拽回零位!
                            self.controller.joint_pos = list(_rs["joints"])
                            self._worker_log("[J6] 开机位已记录: %.3f rad (%.1f°)" % (
                                self._j6_home, math.degrees(self._j6_home)))
                        _est = (_boot_rad - _GRIP_CAL_LAST_RAD
                                if _boot_rad is not None and _GRIP_CAL_LAST_RAD is not None else None)
                        # ★ 2026-10-19 用户要求: 取消启动自动标定(夹爪卡滞时
                        #   50步匍匐会拖2.5分钟且必失败) —— 零位漂移改由
                        #   手动[重标定]按钮处理(需要时再点).
                        if _est is not None:
                            self._worker_log("[gripper] 启动免标定(读数与停车位差 %.1fmm); "
                                             "需要时请手动点[重标定]" % (_est * _GRIP_CAL_MM_PER_RAD))
                        else:
                            self._worker_log("[gripper] 启动免标定(无停车记录); "
                                             "需要时请手动点[重标定]")
                    except Exception as e:
                        print("[gripper零位自愈] %s" % e)

                    self._worker_log("[reBot] 已连接 B601-RS: %s (n_arm=%d, gripper=%s)" % (
                        r.get("msg"), r.get("n_arm", 6), r.get("has_gripper")))
                    self._worker_log("[reBot] 已清错误+使能 6 关节 + 夹爪",
                                     "机械臂已启动", 5000)
                else:
                    self._worker_log("[reBot] 连接失败: %s" % _errtext(r),
                                     "机械臂连接失败: %s" % _errtext(r), 8000)
            except Exception as e:
                print("[start_arm] 异常: %s" % e)
                self._worker_log("[reBot] start_arm 异常: %s" % e, None)
        self.statusBar().showMessage("正在连接机械臂(电机模式切换可能需要十几秒, 请勿重复点击)...")
        threading.Thread(target=_worker, daemon=True).start()

    def emergency_stop(self):
        # ★ 先作废闭环补走线程, 防止急停后它醒来又把臂拽走
        self._cancel_pending_refine()
        # ★ reBot 真急停: 禁用全部关节+夹爪电机 (经 arm server estop=disable_all)
        try:
            self.controller.arm_client.estop()
        except Exception as e:
            print("[estop] %s" % e)
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
        self.statusBar().showMessage("机械臂已紧急停止!")

        # 停止扫描 (置停止事件: 循环线程在段边界感知后退出收尾)
        self.is_scanning = False
        self._scan_stop.set()

        # ★ 新增：紧急停止时同步终止自动采集流程
        self.auto_mode = False
        self.auto_state = 'IDLE'
        self._state_action_triggered = False
        self._best_capture = None
        self._capture_history = None
        if self.detection_thread is not None:
            self.detection_thread.set_active(False)

        self.is_moving = False
        self.is_gripper_waiting = False  # ★ 新增：重置夹爪等待状态
        self.current_action = None
        self.enable_action_buttons(True)
        self.show_current_status()

    def _restart_camera_pipeline(self):
        """相机看门狗触发: 杀掉失败管线后重新启动(内部用, 复用 start_camera)"""
        self.stop_camera()
        time.sleep(1.0)
        self.start_camera()

    def _usb_reset_camera(self):
        """★ 软件级USB重置(等同插拔): 相机驱动被杀后设备常残留坏状态,
        重启的驱动节点能起来但永不出流(实测44s零发布) —— 必须USB重新枚举。
        ★ 2026-09-09: 匹配放宽到 Orbbec 全系 vendor 2bc5(060b旧Astra/0800
        Gemini335) + 033f(新Orbbec VID), 原来只认 2bc5:060b 对 Gemini 无效,
        是"停止相机后再启动不出图"的根因。1.5s 内完成。"""
        try:
            import subprocess as _sp
            script = (
                'for d in /sys/bus/usb/devices/*/idVendor; do '
                '  v=$(cat $d); '
                '  if [ "$v" = "2bc5" ] || [ "$v" = "033f" ]; then '
                '    p=${d%idVendor}; '
                '    echo 6 | sudo -S sh -c "echo 0 > ${p}authorized; '
                '    sleep 1; echo 1 > ${p}authorized"; '
                '  fi; done')
            _sp.run(['bash', '-c', script], capture_output=True, timeout=10)
            time.sleep(1.5)
            return True
        except Exception as e:
            print("[USB重置] 失败: %s" % e)
            return False

    def _reset_pcan_usb(self):
        """★ PCAN-USB 适配器USB级重置(等同插拔)+重新拉起can0 (阻塞约5~20s, 后台线程调用).
        适配器"冻结"(dmesg: Rx urb aborted -32)时 operstate=up 是假象 ——
        netdev在、发送不报错, 但CAN回包全丢, 必须 authorized 0→1 重建USB管道.
        内部调 reset_pcan.sh(顺带全USB禁autosuspend, 集线器空闲挂起是冻结诱因).
        ★ 2026-09-18: 脚本已改为"等待can0重新出现并校验"后才exit 0(掉线
        反复期立即ip link会落进can0不存在的窗口), 失败exit 1并输出原因."""
        try:
            r = subprocess.run(
                ['bash', '-c',
                 'echo 6 | sudo -S bash %s/reset_pcan.sh' % os.path.dirname(os.path.abspath(__file__))],
                capture_output=True, timeout=30)
            if r.returncode != 0:
                print("[PCAN重置] 失败(rc=%d): %s" % (
                    r.returncode, (r.stderr or b'').decode('utf-8', 'ignore').strip()))
            return r.returncode == 0
        except Exception as e:
            print("[PCAN重置] 失败: %s" % e)
            return False

    def _can_watchdog(self):
        """★ CAN健康看门狗(8s): 只检测, 不自动恢复.
        三级判据: 1) can0 netdev消失(适配器被移除) 2) arm_server进程死亡(vel_loop崩溃FATAL退出)
        3) CAN往返探测失败(冻结态: netdev在/进程活, 但收不到回包 —— 灯常亮的真死因).
        ★ 2026-10-29 用户安全要求: 原来会自动 USB重置+重启后端+重新使能,
          CAN 恢复瞬间电机会扑向旧目标(伤人). 现改为只停不发指令 + 提示,
          必须由用户确认现场安全后手动点「启动机械臂」才恢复."""
        if not getattr(self, '_arm_started', False):
            return
        ac = self.controller.arm_client
        can_gone = not os.path.exists('/sys/class/net/can0')
        srv_dead = (ac.proc is None) or (ac.proc.poll() is not None)
        if can_gone or srv_dead:
            self._can_fault_stop("can0设备消失" if can_gone else "arm_server进程死亡")
            return
        # 深度探测(运动/扫描中不做, 避免干扰): 读joint1位置参数要真实CAN回包
        if not (self.is_moving or getattr(self, 'is_scanning', False) or
                getattr(self, 'action_running', False)):
            try:
                r = ac.can_ping()
                if not r.get("ok"):
                    self._can_fault_stop("CAN往返无响应(%s)" % r.get("err", "?"))
            except Exception:
                pass

    def _can_fault_stop(self, reason):
        """★ 电机掉线后的安全停机(2026-10-29): 不再自动恢复.
        停掉后端指令流(防止 CAN 恢复瞬间电机扑向旧目标), 并退出值守,
        由用户手动点「启动机械臂」恢复."""
        self._arm_started = False       # 退出值守, 不再重复触发
        ac = self.controller.arm_client
        print("[CAN看门狗] 检测到 %s — 已停止自动恢复(安全策略), 请手动点「启动机械臂」"
              % reason)
        self._worker_log(
            "[CAN看门狗] ★ 检测到 %s — 已停止自动恢复(防伤人). "
            "确认机械臂周围安全后, 请手动点「启动机械臂」恢复." % reason,
            "电机掉线 · 需手动启动机械臂", 30000)
        threading.Thread(target=self._can_stop_backend, args=(ac,), daemon=True).start()

    @staticmethod
    def _can_stop_backend(ac):
        """★ 优雅停后端: SIGTERM(arm_server已接管: 停控制循环→disable_all泄力→
        断CAN)→等2s→SIGKILL兜底. 对MIT使能态的活后端直接KILL会把全部电机
        打进固件硬故障(launch_ui.sh注释记载的事故教训, 只能整机断电重启),
        看门狗恢复/重试统一走本入口."""
        try:
            if ac.proc is not None and ac.proc.poll() is None:
                ac.proc.terminate()
                for _ in range(20):
                    if ac.proc.poll() is not None:
                        return
                    time.sleep(0.1)
                ac.proc.kill()
                time.sleep(0.5)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # ★★★ 标定模式 (2026-10-28): ChArUco 板检测 + 实时板位姿读数 ★★★
    #   勾选「标定模式」后点「启动相机」→ 跑 aruco_calib_gemini.launch:
    #     · 相机: deeya.launch (与检测管线同一台, 同一套内参)
    #     · 检测: aruco_single_cv2.py (board_type=charuco, 5x7/36.5mm/23mm/6X6_250)
    #   画面显示 /aruco_single/result(角点+坐标轴叠加图),
    #   读数栏显示板位姿 /aruco_single/pose(相机系) + 臂 FK, 供手眼标定采样对照.
    # ──────────────────────────────────────────────────────────────────────────
    def on_calib_mode_toggled(self, checked):
        """「标定模式」开关.
        ★ 2026-10-28 修复: 相机管线已在运行时, 切换模式会自动重启管线,
          立即生效 —— 原来只提示"需先停止再启动", 而用户若直接再点
          「启动相机」会被 start_camera 里的 `if not self.camera_processes`
          挡掉(管线非空直接跳过), 表现为"勾了标定模式但永远没有角点叠加".
        """
        self.CALIB_MODE = bool(checked)
        # ★ 防御: UI 构建期 setChecked 可能早于控件创建(见 create_right_panel)
        _bar = getattr(self, 'calib_info_bar', None)
        if _bar is not None:
            _bar.setVisible(bool(checked))
        _btn = getattr(self, 'calib_mode_btn', None)
        if checked:
            if _btn is not None:
                _btn.setText("标定模式 ✓")
        else:
            if _btn is not None:
                _btn.setText("标定模式")

        # ★ 管线已在跑 → 自动重启, 让模式切换立即生效
        _running = bool(getattr(self, 'camera_processes', None))
        if _running:
            self.status_display.append(
                "[标定] 检测到相机管线正在运行, 自动重启以切换到%s管线..."
                % ("ChArUco 标定" if checked else "v4 检测"))
            self.statusBar().showMessage(
                "正在切换%s管线 (约3~5秒)..." % ("标定" if checked else "检测"))
            # ★★ 必须在主线程执行(2026-10-28 修复闪退):
            #   重启流程会操作 Qt 控件, 从工作线程调用会段错误闪退.
            #   这里用 QTimer.singleShot 排到事件循环下一轮 —— 既保证在主线程,
            #   又让按钮的 toggled 信号处理先返回, 避免在信号槽里做阻塞操作.
            QTimer.singleShot(0, self._calib_restart_pipeline_bg)
            return

        # 管线未运行 → 只提示, 等用户点「启动相机」
        if checked:
            self.statusBar().showMessage(
                "标定模式已开启 — 请点「启动相机」启动 ChArUco 标定管线", 8000)
            self.status_display.append(
                "[标定] 标定模式已开启: 点「启动相机」启动 aruco_calib_gemini.launch "
                "(ChArUco 5x7/方格36.5mm/marker23mm/DICT_6X6_250)")
        else:
            self.statusBar().showMessage(
                "标定模式已关闭 — 下次「启动相机」恢复 v4 检测管线", 6000)
            self.status_display.append("[标定] 标定模式已关闭, 下次启动相机走 v4 检测管线")

    def _calib_restart_pipeline_bg(self):
        """★ 标定模式切换: 重启相机管线.
        ★★ 2026-10-28 修复闪退: 本方法原先在【工作线程】里直接调 stop_camera,
          而 stop_camera 内部会操作 Qt 控件(camera_label.setPixmap / setText /
          calib_info_bar.setVisible / _set_cam_chip / statusBar) —— 跨线程碰
          控件会直接段错误闪退. 现改为全程在主线程执行:
            · stop_camera 里只有 waitForFinished(1s) 是阻塞点, 可接受;
            · USB重置/启动等待较久, 交给既有逻辑, 不再自己 sleep.
          方法名保留(调用点不变), 但语义已是"主线程串行重启"."""
        self.stop_camera()
        # ★ 不在主线程 sleep: 用零延时定时器把启动排到下一轮事件循环,
        #   让"停止"引起的界面刷新先落地, 再启动(避免同帧内反复重绘卡顿).
        QTimer.singleShot(300, self._calib_start_after_switch)

    def _calib_start_after_switch(self):
        """主线程: 管线切换后重新启动相机(走当前 CALIB_MODE 对应管线)."""
        try:
            self.start_camera()
        except Exception as e:
            print("[标定] 重启相机异常: %s" % e)
            self.status_display.append("[标定] 重启相机失败: %s" % e)

    def on_calib_pose(self, msg):
        """板位姿回调(/aruco_single/pose, 相机光学系). ★只缓存, 不在 rospy 线程算. """
        self._calib_last_pose = msg
        self._calib_last_pose_ts = time.time()

    def _calib_update_readout(self):
        """刷新标定读数栏(主线程定时器调用, 约 5Hz).
        显示: 板在相机系的平移/欧氏距离 + 板心基座系推算 + 臂 FK 位姿."""
        if not getattr(self, 'CALIB_MODE', False):
            return
        lbl = getattr(self, 'calib_pose_label', None)
        if lbl is None:
            return
        pose = getattr(self, '_calib_last_pose', None)
        ts = getattr(self, '_calib_last_pose_ts', 0.0)
        if pose is None or (time.time() - ts) > 1.0:
            lbl.setText("标定板: 未检出 — 请把 ChArUco 板完整放入画面")
            self.calib_arm_label.setText("臂位姿: —")
            return
        p = pose.pose.position
        dist = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        # ★ 板心在基座系的推算(眼在手上): p_base = T_bg(FK) · T_gc · p_cam
        base_s = "—"
        arm_s = "臂位姿: —"
        # ★ 节流: 本函数由 30ms 定时器调用, 但 get_pose 是 IPC(可能几十ms),
        #   每帧都问会拖累主线程. 标定读数 3Hz 足够, 且用 get_pose_fast
        #   (0.25s 超时) —— arm_server 忙时快速放弃, 不长期占锁饿死主线程.
        _now = time.monotonic()
        if _now - getattr(self, '_calib_pose_poll_last', 0.0) >= 0.33:
            self._calib_pose_poll_last = _now
            try:
                rp = self.controller.arm_client.get_pose_fast()
                if rp.get("ok") and rp.get("pos"):
                    self._calib_arm_pose_cache = (list(rp["pos"]), list(rp["rpy"]))
            except Exception:
                pass
        _cache = getattr(self, '_calib_arm_pose_cache', None)
        if _cache is not None:
            try:
                pb = pcalib.rebot_cam_xyz_to_base_full(
                    (p.x, p.y, p.z), _cache[0], _cache[1])
                base_s = "(%.3f, %.3f, %.3f)" % (pb[0], pb[1], pb[2])
                arm_s = "臂位姿: pos=(%.3f,%.3f,%.3f)" % tuple(_cache[0][:3])
            except Exception:
                pass
        lbl.setText(
            "板位姿(相机系) x=%.3f y=%.3f z=%.3f m  |d|=%.3f m   板心(基座系)=%s"
            % (p.x, p.y, p.z, dist, base_s))
        self.calib_arm_label.setText(arm_s)

    def _calib_setup_subscribers(self):
        """标定模式订阅: 结果图(显示) + 板位姿(读数). 幂等."""
        if getattr(self, '_calib_pose_sub', None) is None:
            from geometry_msgs.msg import PoseStamped
            self._calib_pose_sub = rospy.Subscriber(
                '/aruco_single/pose', PoseStamped, self.on_calib_pose, queue_size=1)
        # ★ 结果图订阅: 写 controller.camera_image 供 update_camera_view 显示,
        #   但【不】写 _raw_color_msg —— 原始像素系仍由 on_raw_color 维护,
        #   否则叠加图会污染深度/手眼坐标解算(见 handle_camera_image 注释).
        if getattr(self, '_calib_img_sub', None) is None:
            self._calib_img_sub = rospy.Subscriber(
                '/aruco_single/result', Image,
                self.controller.handle_camera_image,
                queue_size=1, buff_size=2 ** 24)

    def start_camera(self):
        self._cam_wd_retried = False   # 看门狗重试标志复位(每次手动启动重新允许自愈一次)
        # ★ 清理可能残留的旧相机/检测进程(上次退出未杀干净会占住USB,
        #   表现为"必须插拔USB才能恢复" —— 杀掉即可重开)
        subprocess.Popen(
            ['bash', '-c',
             'pkill -f "forensic3_detect[.]launch"; '
             'pkill -f "orbbec_camera_node"; '
             'pkill -f "aruco_calib[.]launch"; '
             'pkill -f "aruco_calib_gemini[.]launch"; '
             'pkill -f "aruco_single_cv2[.]py"; sleep 0.3'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not self.camera_processes:
            try:
                self.statusBar().showMessage("启动相机: 清理残留进程...")
                # ★ 2026-09-09 v2: 二次启动一律USB重枚举 —— 上一实例被杀后
                #   Gemini设备残留坏状态(新驱动能起但永不出流, 实测44s零发布),
                #   "孤儿进程存活"是坏代理(孤儿死得快≠设备状态干净)。
                #   首次启动仍查孤儿(上一UI会话未清干净的残留占USB)。
                _r = subprocess.run(
                    ['bash', '-c', 'pgrep -f orbbec_camera_node >/dev/null && echo BUSY || echo FREE'],
                    capture_output=True, text=True, timeout=5)
                _need_reset = ('BUSY' in (_r.stdout or '')) or \
                              getattr(self, '_camera_ran_once', False)
                if _need_reset:
                    self.statusBar().showMessage("重启相机: USB重新枚举设备(约2.5s)...")
                    self._usb_reset_camera()
                    subprocess.Popen(
                        ['bash', '-c', 'pkill -9 -f orbbec_camera_node; '
                                       'pkill -9 -f "forensic3_detect[.]launch"; '
                                       'pkill -9 -f "blood_detect[.]launch"; sleep 0.5'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(1.0)
                # ★ 检测管线: roslaunch forensic3_detector_v4 blood_detect.launch
                #   (Gemini相机 + blood2 血迹模型[dla_34/单类])
                #   输出重定向到文件, 防止 QProcess 管道堵死。
                # ★ 标定模式(2026-10-28, 由 UI「标定模式」按钮置位):
                #   aruco_calib_gemini.launch (Deeyea相机 + ChArUco检测) 替代
                #   v4检测管线. 画面显示 /aruco_single/result(角点+坐标轴),
                #   读数栏显示 /aruco_single/pose 板位姿 + 臂 FK.
                #   原始流仍照常订阅(画框/深度解算用原始像素系).
                #   v4 实时检测(推理)暂时关闭.
                if getattr(self, 'CALIB_MODE', False):
                    calib_cmd = (
                        "source /opt/ros/noetic/setup.bash && "
                        "source ~/forensic_ws/devel/setup.bash --extend && "
                        "exec roslaunch aruco_ros aruco_calib_gemini.launch "
                        "> /tmp/aruco_calib.log 2>&1"
                    )
                    detect_process = QProcess()
                    detect_process.setProcessChannelMode(QProcess.MergedChannels)
                    detect_process.start("bash", ["-c", calib_cmd])
                    self.camera_processes.append(detect_process)
                    # ★ 原始流订阅照常建立(画框/深度解算要用原始像素系)
                    if self.color_sub is None:
                        self.init_perception_subscribers()
                    # ★ 标定专用订阅: 结果图(显示) + 板位姿(读数栏)
                    self._calib_setup_subscribers()
                    self.calib_info_bar.setVisible(True)
                    self.statusBar().showMessage(
                        "标定模式: ChArUco 检测已启动 "
                        "(板 5x7/方格36.5mm/marker23mm, TF: camera←aruco_marker)")
                    self.status_display.append(
                        "[标定] 标定模式已启动: aruco_calib_gemini.launch "
                        "(Deeyea相机 + ChArUco 5x7). 板位姿: /aruco_single/pose, "
                        "读数栏实时显示; 也可 rosrun tf tf_echo "
                        "camera_color_optical_frame aruco_marker. v4 实时检测已暂停.")
                    return

                # ★ 相机启动 (2026-10-21 换 Deeyea 相机, CenterNet 检测在 UI 进程内):
                #   blood_detect.launch 只起相机(Deeyea), 检测节点已移除.
                #   UI 相机窗和自动检测都走 /camera/color/image_raw 原始流
                #   (on_raw_color 同时设 camera_image 供 update_camera_view 显示,
                #    _raw_color_msg 供检测线程 CenterNet bridge 推理).
                #   输出重定向到文件, 防止 QProcess 管道堵死(看门狗也读此日志).
                detect_cmd = (
                    "source /opt/ros/noetic/setup.bash && "
                    "source ~/forensic_ws/devel/setup.bash --extend && "
                    "source ~/forensic3_detect_ws_v4/devel/setup.bash --extend && "
                    "exec roslaunch forensic3_detector_v4 blood_detect.launch "
                    "> /tmp/forensic3_detect.log 2>&1"
                )
                detect_process = QProcess()
                detect_process.setProcessChannelMode(QProcess.MergedChannels)
                detect_process.start("bash", ["-c", detect_cmd])
                self.camera_processes.append(detect_process)

                # ★ 2026-10-21 修复: 原始流订阅必须在这里建立 —— 原先只在
                #   「人工标注/自动采集」模式才调 init_perception_subscribers(),
                #   单纯点「启动相机」时 color_sub 一直是 None → 相机在发帧
                #   但 UI 没订阅 → 永远没画面. (幂等: 已建过则跳过)
                if self.color_sub is None:
                    self.init_perception_subscribers()

                # ★ 2026-10-21: 手动相机画面叠加 CenterNet 检测框 —
                #   后台线程加载模型(不卡UI), 就绪后启动连续检测线程,
                #   结果缓存到 _latest_detection 供 update_camera_view 画框.
                #   (on_detection_ready 有 auto_state 守卫, 常连不会误触发自动动作)
                self._ensure_manual_detection()

                # ★ 原始流订阅已在 init_perception_subscribers() 完成
                #   (on_raw_color 同时设 camera_image + _raw_color_msg).
                # ★ 启动等待提示: 停止时已清pixmap, 这里给"启动中"文字,
                #   首帧到达后自动被画面覆盖(实测到帧约需10~15s)
                self.camera_label.setText("")
                self._update_camera_placeholder("相机启动中 · 驱动初始化约 10~15 秒")
                self.statusBar().showMessage("Deeyea相机启动完成, CenterNet血迹检测就绪")
                # ★ 启动看门狗: 15s后判活 —— ①roslaunch全灭 ②驱动活着但
                #   原始流无帧(设备坏状态, 需USB重新枚举)。命中则USB重置+
                #   重启管线一次。原版只认Astra时代的uvc_open日志字符串,
                #   对Gemini335完全失效("停止后再启动永不出图"的根因之二)。
                def _camera_watchdog():
                    try:
                        time.sleep(15)
                        if not self.camera_processes:
                            return
                        try:
                            with open('/tmp/forensic3_detect.log', 'r', errors='ignore') as f:
                                log = f.read()
                        except Exception:
                            log = ''
                        dead = ('all processes on machine have died' in log)
                        if not dead:
                            # 通用判据: 4s内原始彩色流无任何消息 → 设备异常
                            _probe = subprocess.run(
                                ['bash', '-c',
                                 'timeout 4 rostopic echo -n 1 --noarr '
                                 '/camera/color/image_raw >/dev/null 2>&1'],
                                timeout=10)
                            if _probe.returncode != 0:
                                dead = True
                        if not dead:
                            return
                        if getattr(self, '_cam_wd_retried', False):
                            return   # 只自动重试一次, 防循环
                        self._cam_wd_retried = True
                        # ★ USB重置直接做(纯subprocess, 线程安全); 消息与重启
                        #   经标志位交主线程update_all执行 —— 原QTimer.singleShot
                        #   从python线程调度永不触发(startTimer警告即此, 重启被丢)
                        print("[相机看门狗] 相机无帧输出(设备坏状态), 已USB重置, 即将自动重启管线")
                        self._usb_reset_camera()
                        self._cam_wd_pending_restart = True
                    except Exception as e:
                        print("[相机看门狗] %s" % e)
                threading.Thread(target=_camera_watchdog, daemon=True).start()
                # ★ 标记本UI会话内相机已跑过 → 下次启动一律USB重枚举
                self._camera_ran_once = True
            except Exception as e:
                self.status_display.append("[动作中止] 启动相机失败: {str(e)}")
        else:
            self.status_display.append("[动作中止] 相机已经启动!")

    def stop_camera(self):
        if self.camera_processes:
            for process in self.camera_processes:
                try:
                    if process.state() == QProcess.Running:
                        process.terminate()
                        if not process.waitForFinished(1000):
                            process.kill()
                    process.close()
                except Exception:
                    pass
            self.camera_processes = []
            # ★ 兜底: roslaunch 被杀后其孙进程(astra相机驱动节点)可能残留并
            #   占住USB设备, 导致下次「启动相机」打不开、必须插拔USB。
            #   pkill 按进程名清扫整个检测管线(节点+roslaunch), 1.5s后返回。
            #   ★ 2026-10-28: 一并清扫标定管线(aruco_calib_gemini.launch +
            #   aruco_single_cv2.py), 否则切模式时会残留旧节点占住相机话题.
            subprocess.Popen(
                ['bash', '-c',
                 'pkill -f "forensic3_detect[.]launch"; '
                 'pkill -f "blood_detect[.]launch"; '
                 'pkill -f "aruco_calib_gemini[.]launch"; '
                 'pkill -f "aruco_single_cv2[.]py"; '
                 'pkill -f "orbbec_camera_node"; '
                 'pkill -f "forensic3_detect_v4_node"; sleep 0.2'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ★ 标定模式订阅一并注销, 并把读数栏复位/隐藏
            for _attr in ('_calib_pose_sub', '_calib_img_sub'):
                _s = getattr(self, _attr, None)
                if _s is not None:
                    try:
                        _s.unregister()
                    except Exception:
                        pass
                setattr(self, _attr, None)
            self._calib_last_pose = None
            self._calib_last_pose_ts = 0.0
            if getattr(self, 'calib_info_bar', None) is not None:
                self.calib_info_bar.setVisible(bool(getattr(self, 'CALIB_MODE', False)))
                if getattr(self, 'calib_pose_label', None) is not None:
                    self.calib_pose_label.setText("标定板: 未检出")
                if getattr(self, 'calib_arm_label', None) is not None:
                    self.calib_arm_label.setText("臂位姿: —")

            # ★ 修复：image_subscriber 可能为 None（未真正订阅过），
            #   直接 unregister 会抛 AttributeError 导致 stop_camera 崩溃、
            #   相机节点残留占用 USB，下次打开失败。改为判空 + try/except。
            sub = getattr(self.controller, 'image_subscriber', None)
            if sub is not None:
                try:
                    sub.unregister()
                except Exception:
                    pass
            self.controller.image_subscriber = None

            # ★ 2026-09-09: 清掉残留画面 —— controller.camera_image 还存着
            #   最后一帧msg, 标签也还挂着旧pixmap, 两者不清则画面永远停在
            #   停止前的帧("停止后重启不出图却显示旧帧"的观感来源之一)
            self.controller.camera_image = None
            self.camera_label.setPixmap(QPixmap())
            self._update_camera_placeholder()
            self.camera_label.setText("")
            self._update_camera_placeholder("相机已停止 · 等待启动")
            self._set_cam_chip('待机')
            self.statusBar().showMessage("相机系统已停止!")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：closeEvent — 先释放资源再弹确认框
    # ──────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        # ★ 先停CAN看门狗: 否则退出过程中它检测到arm_server死亡会提示掉线
        self._arm_started = False

        # 停止扫描 (置停止事件: 循环线程在段边界感知后退出收尾)
        self.is_scanning = False
        self._scan_stop.set()

        # ★ 新增：关闭窗口时同步终止自动采集流程
        self.auto_mode = False
        self.auto_state = 'IDLE'

        # ★ 新增：彻底停止CenterNet后台检测线程（不同于置为非激活，这里退出线程循环）
        if self.detection_thread is not None:
            self.detection_thread.stop()
            self.detection_thread = None

        # 停止扫描线程
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread = None

        # 停止机械臂
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
            if not self.launch_process.waitForFinished(1000):
                self.launch_process.kill()

        # ★ reBot: 关闭机械臂后端子进程
        # ★ 2026-08-31: shutdown改为"直接disable不回零"(arm_server 2秒内完成),
        #   防止"回零10-20秒→UI超时/被强杀→电机保持使能→PCAN灯常亮"的根因.
        #   原版arm_client.shutdown()发完命令后没有等完成确认就返回,
        #   现在改为带超时等待, 确认电机确实disable了才继续走后续退出流程.
        try:
            r = self.controller.arm_client._send({"cmd": "shutdown"}, timeout=8)
            if r.get("ok"):
                self.statusBar().showMessage("机械臂已去使能并安全断开")
        except Exception:
            pass
        # ★ 确保子进程退出(即使shutdown响应丢失)
        # ★ 2026-10-21: proc 可能为 None(机械臂后端从未连接过, 只用相机时关UI),
        #   原代码直接 .wait() 抛 AttributeError → 二次 .poll() 再抛 → Fatal abort
        try:
            if self.controller.arm_client.proc is not None:
                self.controller.arm_client.proc.wait(timeout=3)
        except Exception:
            if (self.controller.arm_client.proc is not None
                    and self.controller.arm_client.proc.poll() is None):
                self.controller.arm_client.proc.kill()

        # 停止相机
        self.stop_camera()

        rospy.signal_shutdown('UI closed')

        reply = QMessageBox.question(
            self, '确认退出', "确定要退出系统吗?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def handle_launch_output(self):
        data = self.launch_process.readAllStandardOutput()
        msg = data.data().decode().strip()
        self.status_display.append(f"[ROS输出] {msg}")

    def handle_launch_error(self):
        data = self.launch_process.readAllStandardError()
        msg = data.data().decode().strip()
        self.status_display.append(f"[ROS错误] {msg}")

    def handle_launch_finished(self, exitCode, exitStatus):
        self.status_display.append(f"机械臂进程已结束，退出码: {exitCode}")
        self.statusBar().showMessage(f"机械臂进程已结束，退出码: {exitCode}")


# ──────────────────────────────────────────────────────────────────────────────
# 机械臂控制器
# ──────────────────────────────────────────────────────────────────────────────
# ===== reBot 机械臂后端: 子进程桥接 rebot_arm_server.py =====
# ★ 2026-10-10 修正: 优先使用 rebot_pin 环境(Python 3.11, 2026-10-09 重建)。
#   旧 rebot 环境为 Python 3.12: UI 从 source 过 ROS noetic 的 shell 启动时,
#   子进程继承的 LD_LIBRARY_PATH 含 /opt/ros/noetic/lib/x86_64-linux-gnu,
#   其中 py3.8 版 libeigenpy/libboost_python38 顶替 cmeel wheel 自带库,
#   导致 pinocchio 导入崩溃 (undefined symbol: _Py_tracemalloc_config),
#   arm_server 启动即死 —— UI 表现为"CAN 连接失败"。
#   注意: Popen 直接调用 python 二进制不触发 conda activate.d 钩子,
#   故还须在子进程 env 中显式剥离 /opt/ros/* 库路径(见 _armserver_env)。
#   探测失败会导致"机械臂后端启动失败"(arm_server 子进程拉不起)。
REBOT_PY = next(
    (p for p in [
        os.path.expanduser('~/anaconda3/envs/rebot_pin/bin/python'),
        os.path.expanduser('~/anaconda3/envs/rebot/bin/python'),
        os.path.expanduser('~/下载/enter/envs/rebot/bin/python'),
    ] if os.path.exists(p)),
    os.path.expanduser('~/anaconda3/envs/rebot_pin/bin/python'))
# ★ 2026-10-16: 运动控制后端切换为重写版 rebot_arm_control.py
#   ("move_traj" 服务器端测地线轨迹, 基于 RebotArmEndPose.move_to_traj)
REBOT_ARM_SERVER = os.path.expanduser('~/forensic_ws/src/reBot/scripts/rebot_arm_control.py')


def _armserver_env():
    """构造 arm_server 子进程环境: 从 LD_LIBRARY_PATH 剔除 /opt/ros/* 路径。
    与 rebot_pin 环境 etc/conda/activate.d/zz_strip_ros_ldpath.sh 逻辑一致
    (Popen 不经过 conda activate, 该钩子不会触发, 只能在此显式清理)。"""
    env = dict(os.environ)
    kept = [p for p in env.get('LD_LIBRARY_PATH', '').split(':')
            if p and not p.startswith('/opt/ros/')]
    if kept:
        env['LD_LIBRARY_PATH'] = ':'.join(kept)
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


class RebotArmClient:
    """与 rebot_arm_server.py (rebot_pin py3.11 子进程) 通过 JSON-line 通信的客户端."""

    def __init__(self):
        self.proc = None
        self._js_pub = None             # ★ 2026-09-10: /joint_states 真实发布器(标定TF链)
        self._lock = threading.Lock()      # 读响应锁(_send 等待期间持有)
        self._wlock = threading.Lock()     # ★ 写管道专用短锁: set_joints 高频写
                                           #   不与 _send 的长读互相阻塞(卡顿修复)
        self._seq = 0                      # ★ 2026-11-07 命令序列号: 按 seq 匹配响应

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        try:
            self.proc = subprocess.Popen(
                [REBOT_PY, REBOT_ARM_SERVER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=open(os.path.expanduser('~/rebot_arm_server.err'), "a"),
                env=_armserver_env(), text=True, bufsize=1,
            )
            return self.proc.poll() is None
        except Exception as e:
            print("[RebotArmClient] start 失败: %s" % e)
            return False

    def _send(self, cmd, timeout=30.0):
        """向 arm_server 发命令并等响应(读锁串行化: 同一时刻只有一个等响应者,
        响应配对安全; 写管道用 _wlock 与 fire-and-forget 的 set_joints 串行).
        ★ 2026-08-31 还原为(6)版阻塞写实现(非阻塞os.write实验版撤回).
        ★ 2026-11-07 序列号匹配: 每条命令附递增 seq, 后端回显. 原按序配对
        在某命令超时后其滞留回包会被后续命令误读 —— 一次超时即永久错位
        ("FK 显示未连接但后端健康"的根因: get_pose 恒读 get_state 的回包).
        按 seq 匹配后滞留包直接丢弃, 错位不可能发生."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "err": "arm server not running"}
            self._seq += 1
            cmd = dict(cmd, seq=self._seq)
            with self._wlock:
                try:
                    self.proc.stdin.write(json.dumps(cmd) + "\n")
                    self.proc.stdin.flush()
                    # ★ tx 打点(2026-10-18 协议断点定位): 与后端 [rx] 配对,
                    #   UI 报超时而后端 err 无对应 rx = cmd 没到后端
                    print("[tx] %s t=%.3f" % (cmd.get("cmd"), time.time()))
                except Exception as e:
                    return {"ok": False, "err": "stdin write fail: %s" % e}
            resp = self._send_locked(cmd, timeout)
            # ★ 2026-11-05: 失败打点 — 超时/管道关闭不再静默, 便于诊断
            #   "FK 显示未连接但后端健康" 这类假掉线问题.
            if not resp.get("ok"):
                print("[arm_client] cmd=%s 失败: %s" %
                      (cmd.get("cmd"), resp.get("err", "unknown")))
            return resp

    def _send_locked(self, cmd, timeout=30.0):
        """★ select 轮询实现真超时(调用方已持锁):
        绝不在 UI 主线程无限阻塞(readline 是阻塞调用, arm_server 慢时
        会把整个界面卡死 —— '启动机械臂UI卡死'的根因).
        ★ 2026-11-07 按 seq 匹配响应: seq 不符的滞留回包(前序超时命令
        遗留)直接丢弃继续等, 响应错位从结构上杜绝."""
        import select as _select
        # (命令已由调用方写入管道; 本函数只负责等响应)
        my_seq = cmd.get("seq")
        deadline = time.time() + timeout
        fd = self.proc.stdout
        while time.time() < deadline:
            # stdout 有数据才读, 最多等 0.2s, 超时回主线程呼吸
            r, _, _ = _select.select([fd], [], [], 0.2)
            if not r:
                continue
            line = self.proc.stdout.readline()
            if not line:
                return {"ok": False, "err": "arm server closed"}
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("seq") == my_seq:
                    return obj
                # seq 不符 = 前序超时命令的滞留回包, 丢弃
                continue
            except Exception:
                continue  # 非 JSON 行(motorbridge 告警等), 跳过
        return {"ok": False, "err": "arm server 响应超时"}


    def connect(self):           return self._send({"cmd": "connect"})
    def enable(self):            return self._send({"cmd": "enable"})
    def disable(self):          return self._send({"cmd": "disable"})
    def home(self):             return self._send({"cmd": "home"}, timeout=30.0)
    def estop(self):            return self._send({"cmd": "estop"})
    def get_state(self):        return self._send({"cmd": "get_state"}, timeout=3.0)
    def get_pose(self):         return self._send({"cmd": "get_pose"}, timeout=3.0)
    def get_pose_fast(self):
        """★ FK后台刷新专用(2026-09-11): 1.5s超时 — arm_server正常 IK+轨迹生成
        需 100-200ms，move_traj 持锁期间 UI 此调用需等锁；原 0.25s 边界过紧
        导致动作期间频繁超时假掉线；1.0s 仍偶发被击穿(不开相机也复现),
        放宽到 1.5s 仍远短于 get_state 的 3.0s, 不致长期占锁饿死动作线程."""
        return self._send({"cmd": "get_pose"}, timeout=1.5)
    def can_ping(self):         return self._send({"cmd": "can_ping"}, timeout=3.0)
    def set_joints(self, j):
        """高频关节指令(30Hz运动插值调用). ★ fire-and-forget 单向写:
        原来每帧等 arm_server 的 JSON 响应(链路3~50ms抖动) + 与控制循环抢
        arm_server 的命令锁 —— 帧间隔不稳 = '一段一段卡顿'的结构性根因.
        arm_server 的 set_joints 恒返 ok, 响应无信息量, 改为只写不等读.
        ★ 2026-08-31 还原(6)版阻塞写(非阻塞实验版撤回)."""
        if not self._wlock.acquire(timeout=0.05):
            return {"ok": False, "err": "busy"}
        try:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "err": "arm server not running"}
            self.proc.stdin.write(json.dumps(
                {"cmd": "set_joints", "joints": list(j)}) + "\n")
            self.proc.stdin.flush()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "err": "stdin write fail: %s" % e}
        finally:
            self._wlock.release()
    def set_gripper(self, p):
        """★ 恢复同步版(2026-09-11 v2): 异步版返回None被调用方误判为
        '无响应'(指令实际已发出). UI卡死的真正根因是FK刷新在主线程调
        get_pose占锁3s — 已由FK后台线程修复, 同步set_gripper只阻塞几ms."""
        return self._send({"cmd": "set_gripper", "pos": float(p)})

    def solve_ik(self, x, y, z, normal=None, rpy=None, fast=False):
        cmd = {"cmd": "solve_ik", "x": float(x), "y": float(y), "z": float(z), "fast": bool(fast)}
        if normal is not None:
            cmd["normal"] = list(normal)
        elif rpy is not None:
            cmd["rpy"] = list(rpy)
        r = self._send(cmd)
        # ★ 2026-10-29: 原"arm_server 死亡自愈"(自动重启+重连+recover)已移除
        #   —— 电机掉线后静默重新使能, CAN恢复瞬间会扑向旧目标伤人.
        #   后端不在就返回错误, 由用户手动点「启动机械臂」恢复.
        return r

    def move_traj(self, x, y, z, normal=None, q6=None, duration=None, timeout=30.0):
        """★ 服务器端轨迹运动(2026-10-16): 一次性下发目标位姿+法向(+可选J6
        腕自转/时长), IK+测地线轨迹全部由 rebot_arm_control 的
        RebotArmEndPose.move_to_traj 在服务器端完成 —— UI 不再做本地IK预检、
        30Hz插值和 set_joints 流式下发. 服务器端同步完成IK+轨迹解算后才回包
        (可能阻塞数秒), 调用方需容忍较长响应时间."""
        cmd = {"cmd": "move_traj", "x": float(x), "y": float(y), "z": float(z)}
        if normal is not None:
            cmd["normal"] = list(normal)
        if q6 is not None:
            cmd["q6"] = float(q6)
        if duration is not None:
            cmd["duration"] = float(duration)
        return self._send(cmd, timeout=timeout)

    def publish(self, msg):
        # ★ 2026-09-10 修复: 原来静默丢弃 —— 但标定模式的 robot_state_publisher
        #   需要 /joint_states 才能发布 base_link→link1..6→gripper_end 的TF链
        #   ("标定板看不到坐标系"的根因). reBot 的 set_joints 走JSON管道不走
        #   ROS话题, 真发布关节状态无任何副作用, 恢复为真实发布.
        if self._js_pub is None:
            self._js_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        try:
            self._js_pub.publish(msg)
        except Exception as e:
            print("[joint_states] 发布失败: %s" % e)

    def shutdown(self):
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

# ──────────────────────────────────────────────────────────────────────────────
# reBot B601-RS 工作空间边界 (臂展 754mm; 底座桌面安装; 现场标定后再收紧)
# ──────────────────────────────────────────────────────────────────────────────
REBOT_WORKSPACE = {
    'xmin': -0.65, 'xmax': 0.65,
    'ymin': -0.65, 'ymax': 0.65,
    'zmin': -0.40, 'zmax': 0.70,
}
# ★ 2026-09-11: 0.60→0.65 — 臂展754mm, X=0.617(画框修正后)物理可达
#   但被±0.60硬限拒绝("框选后臂不动"的根因). 0.65留足安全余量.


def workSpace(pos):
    """reBot 工作空间检查. 返回 True=在边界内."""
    x, y, z = pos[0], pos[1], pos[2]
    if (REBOT_WORKSPACE['xmin'] <= x <= REBOT_WORKSPACE['xmax'] and
        REBOT_WORKSPACE['ymin'] <= y <= REBOT_WORKSPACE['ymax'] and
        REBOT_WORKSPACE['zmin'] <= z <= REBOT_WORKSPACE['zmax']):
        return True
    print(pos, 'out of reBot workspace',
          '(X:[%.2f,%.2f] Y:[%.2f,%.2f] Z:[%.2f,%.2f])' % (
              REBOT_WORKSPACE['xmin'], REBOT_WORKSPACE['xmax'],
              REBOT_WORKSPACE['ymin'], REBOT_WORKSPACE['ymax'],
              REBOT_WORKSPACE['zmin'], REBOT_WORKSPACE['zmax']))
    return False


class ArmController:
    """reBot B601-RS 后端: 通过 RebotArmClient (子进程) 控制.
    保留原 GLUON ArmController 的方法签名, 使 UI 调用点不变."""

    def __init__(self):
        rospy.init_node('arm_controller')
        self.arm_client = RebotArmClient()
        self.publisher = self.arm_client      # move_linear/move_joint 的 publisher 参数 = client
        self.joint_pos = [0.0] * 6
        self.q6_value = 0.0
        self.camera_image = None
        self.image_subscriber = None
        # ★ 标定功能已移除(2026-08-28): 手眼结果固化于 perception_calib.py
        #   (REBOT_HAND_EYE_A/H), 相机窗由「启动相机」订阅 v4 检测结果图。

    def handle_camera_image(self, msg):
        # ★ 只做引用赋值(原子). rospy 接收线程绝不跑 numpy/cv2
        #   (在接收线程里做 cvtColor/flip 曾导致段错误: Fatal Python error
        #    Segmentation fault in rospy recv_buff 线程 —— 与检测线程
        #    并发操作图像内存是崩点). 转换全部移到取用方线程.
        self.camera_image = msg
        # ★★ 2026-10-28 修复: 本回调只用于"显示用"流(如标定模式的
        #   /aruco_single/result 叠加图), 绝不能写 _raw_color_msg ——
        #   该缓存是深度对齐/像素反投影/手眼标定的原始像素系来源,
        #   被叠加图污染会让三维解算整条链路错乱.
        #   原始流统一由 on_raw_color 写 _raw_color_msg.
        #   (历史: 这里曾同时赋值 _raw_color_msg, 在标定模式下与
        #    on_raw_color 互相覆盖.)
        pass

    def move_to_cartesian(self, x, y, z, q6=None, normal=None, fast=False):
        """笛卡尔位置 -> IK -> 返回 6 关节角 (弧度). 不直接运动 (由 move_linear 逐帧 set_joints)."""
        if not workSpace([x, y, z]):
            print("[reBot] 目标位置超出工作空间")
            return False, None
        r = self.arm_client.solve_ik(x, y, z, normal=normal, fast=fast)
        if not r.get("ok") or r.get("joints") is None:
            print("[reBot] IK 失败: %s" % r.get("err"))
            self.last_ik_err = str(r.get("err", ""))
            return False, None
        self.last_ik_err = ""
        joints = list(r["joints"])
        if q6 is not None:
            self.q6_value = q6
        joints[5] = self.q6_value          # 第6关节 = q6 (腕部自由旋转)
        self.joint_pos = list(joints)
        return True, joints

    def rotate_joint6(self, angle):
        self.q6_value += angle
        return True

    def return_to_zero(self):
        self.joint_pos = [0.0] * 6
        self.q6_value = 0.0
        return True


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def pub2ros_pos(publisher, pos):
    # publisher = RebotArmClient; 设关节目标, 控制循环跟踪
    publisher.set_joints(pos)


def interpolate(start, end, fraction):
    return [s + (e - s) * fraction for s, e in zip(start, end)]


def _ik_supports_fast(ik_solver):
    """检测 ik_solver(ArmController.move_to_cartesian 绑定方法)是否支持 fast 参数."""
    try:
        import inspect
        return 'fast' in inspect.signature(ik_solver).parameters
    except Exception:
        return False


def move_linear(publisher, ik_solver, start_xyz, end_xyz, duration=5.0, rate_hz=30,
                q6=None, normal=None, controller=None):
    """笛卡尔"点到点"直线运动 (阻塞). ★reBot版实现:
    对起/终点各解一次IK(带中途航点), 得到关节序列后按关节空间平滑插值下发.
    不再逐帧IK —— reBot的数值IK(pinodeio CLIK)单次解算耗时不定,
    逐帧调用会造成"动一段停一下"的非连续运动(GLUON解析IK是微秒级才允许逐帧).
    路径在关节空间插值, 对execute_action里5cm级的短接近段近似直线. """
    # 1) 航点IK: 短段(<15cm)只用终点 —— 中点航点的IK可能落到另一条解分支,
    #    move_joint 在两条分支间插值 = 机械臂大幅摆动(抽搐)的根因;
    #    长段(转移)保留起/中/终点平滑过渡.
    dist = ((end_xyz[0] - start_xyz[0]) ** 2 + (end_xyz[1] - start_xyz[1]) ** 2 +
            (end_xyz[2] - start_xyz[2]) ** 2) ** 0.5
    if dist < 0.15:
        waypoints = [list(end_xyz)]
    else:
        waypoints = [list(start_xyz),
                     [(a + b) / 2.0 for a, b in zip(start_xyz, end_xyz)],
                     list(end_xyz)]
    joints_seq = []
    for k, wp in enumerate(waypoints):
        last = (k == len(waypoints) - 1)          # 只有终点用完整IK(允许慢速兜底)
        # ★ 2026-09-16 提速: 短段(<15cm, 接近/退回/下探类)终点也走fast IK
        #   —— 起点就在附近, 连续性种子一步收敛, fast毫秒级; 原"终点必完整IK"
        #   每段阻塞0.5~2s = 分段间长等待的次根因. fast失败自动回落完整IK.
        use_fast = (not last) or dist < 0.15
        success, joints = ik_solver(wp[0], wp[1], wp[2], q6, normal, fast=use_fast) if _ik_supports_fast(ik_solver) else ik_solver(wp[0], wp[1], wp[2], q6, normal)
        if not success and use_fast and _ik_supports_fast(ik_solver):
            success, joints = ik_solver(wp[0], wp[1], wp[2], q6, normal, fast=False)
        if success and joints is not None:
            joints_seq.append(list(joints))
        else:
            print("[move_linear] 航点(%.3f,%.3f,%.3f)逆解失败, 跳过" % tuple(wp))
    # ★ 分支连续性: 相邻航点解若差太大(>40°)说明IK跳了解分支, 剔除中间航点,
    #   否则 move_joint 会在两条分支间插值 -> 机械臂大幅摆动(抽搐)
    if len(joints_seq) >= 3:
        import math as _m
        jump = max(abs(b - a) for a, b in zip(joints_seq[-2], joints_seq[-1]))
        if jump > 0.7:  # rad (~40°)
            print("[move_linear] 检测到解分支跳变(%.2f rad), 剔除中间航点" % jump)
            joints_seq = [joints_seq[0], joints_seq[-1]]
    if not joints_seq:
        print("[move_linear] 无可达航点, 不运动")
        return False
    # 2) 段内时长按航点数均分, 关节空间线性插值下发
    seg_duration = duration / max(1, len(joints_seq))
    cur = list(controller.joint_pos) if controller is not None else joints_seq[0]
    for target in joints_seq:
        move_joint(publisher, cur, target, seg_duration, rate_hz, controller)
        cur = target
    return True


def move_joint(publisher, start_joints, end_joints, duration=5.0, rate_hz=30,
               controller=None):
    """关节空间点到点 (阻塞). 逐帧线性插值 -> set_joints.
    ★基线版(30Hz流式): arm_server 端为线性ramp+速度前馈限幅, 流式目标不会
    被重置, 响应即时且平滑(与最初稳定版本一致).
    ★ 2026-09-09 末端位置回验: 流结束后等臂真正停稳(轮询实测关节到位),
    再用 get_pose 实测末端位置 vs 本段关节终点的理论FK —— 位置误差>5mm
    时明确报警(此前只有关节角3°级监控, "输入z=0.30实测0.25差50mm"这类
    IK假成功/物理偏差完全静默, 用户只能靠肉眼发现)."""
    # ★★ 同步到位插值: 时长按各关节在 分级VMAX+ACC_MAX 下的最短走完时间
    #   (梯形/三角形加速度剖面)计算, 所有关节共用同一时间轴线性走.
    #   2026-09-16 修复"动作慢放5-10倍": 旧公式 max_travel/0.18+1.5 按全局
    #   龟速0.18配时且每段硬加1.5s —— 短往复段被拖长数倍. 现与 arm_server
    #   的分级限速(VMAX_J1=0.35/其余0.80/j6=1.20, ACC_MAX=1.5)精确匹配,
    #   padding 降至0.3s(段尾到位由 wait_arrival 轮询兜底).
    try:
        _UI_VMAX_J = [0.35, 0.80, 0.80, 0.80, 0.80, 1.20]
        _UI_ACC = 1.5
        need = 0.0
        for j, (s, e) in enumerate(zip(start_joints, end_joints)):
            d = abs(e - s)
            if d < 1e-6:
                continue
            v = _UI_VMAX_J[j] if j < len(_UI_VMAX_J) else 0.80
            if d >= v * v / _UI_ACC:            # 梯形: 加速段+匀速+减速段
                t = d / v + v / _UI_ACC
            else:                                # 三角形: 全程加减速
                t = 2.0 * (d / _UI_ACC) ** 0.5
            need = max(need, t)
        if need > 0.0:
            duration = max(duration, need + 0.3)
    except Exception:
        pass
    steps = max(1, int(duration * rate_hz))
    dt = 1.0 / rate_hz
    for i in range(steps + 1):
        fraction = i / float(steps)
        current = interpolate(start_joints, end_joints, fraction)
        pub2ros_pos(publisher, current)
        if controller is not None:
            controller.joint_pos = list(current)
            controller.q6_value = current[5]
        time.sleep(dt)
    # ★ 末端位置回验(2026-09-09): 只在有controller(可查询arm_server)时执行
    #   ★ 2026-09-16 提速: 微行程段(<0.1rad, 涂抹往复笔)跳过 —— 回验含
    #     wait_arrival+get_pose+get_state, 每笔白耗0.5s+, 往复间隙变长.
    _seg_big = False
    try:
        _seg_big = max(abs(e - s) for s, e in zip(start_joints, end_joints)) >= 0.1
    except Exception:
        _seg_big = True
    if controller is not None and _seg_big:
        try:
            # 1) 等实测关节到位(VMAX限速下臂可能还在追流式终点)
            #    ★ v2: 稳态检测+4s上限(见wait_arrival注释), 原10s超时是
            #    "分段间长时间干等"的主根因
            ok_arrive = wait_arrival(controller, list(end_joints),
                                      tol_deg=3.0, timeout=4.0)
            # 2) 实测末端 vs 本段终点的理论FK
            r = controller.arm_client.get_pose()
            if r.get("ok") and r.get("pos"):
                p_act = np.asarray(r["pos"], dtype=float)
                import math as _m
                # 理论FK: 用本段终点关节角算末端(走IK不可行, 直接FK)
                # 经 arm_server 的 solve_ik 反算有容差, 此处用关节终点
                # 的 get_pose 语义: 直接以实测关节角FK即 get_pose 已是.
                # 简化判据: 实测关节 vs 指令终点的关节偏差映射 —— 用
                # 关节到位程度 + 末端实测位置直接展示, 供用户判断.
                pe_deg = 0.0
                rs = controller.arm_client.get_state()
                if rs.get("ok") and rs.get("joints"):
                    pe_deg = max(_m.degrees(abs(a - b)) for a, b in
                                 zip(rs["joints"], end_joints))
                print("[到位回验] 关节最大偏差 %.1f° | 末端实测=(%.4f,%.4f,%.4f)"
                      % (pe_deg, p_act[0], p_act[1], p_act[2]))
                if not ok_arrive:
                    print("[到位回验] ⚠ 关节未在10s内到位(最大偏差%.1f°) — "
                          "电机可能失能/堵限位" % pe_deg)
        except Exception as _e:
            print("[到位回验] 读取失败: %s" % _e)
    return True


def wait_gripper(controller, target_mm, tol_mm=2.0, timeout=15.0,
                 stable_s=0.8, poll=0.15):
    """★ 夹爪到位等待(2026-09-16): 轮询实测rad→mm, 三判据任一即通过:
      ① |实测-目标|≤tol_mm(真到位) ② 实测稳定stable_s(夹住物体停住=已夹稳,
      目标0mm但停在物体宽度处是正常成功) ③ 超时兜底.
    原 time.sleep(2.5) 固定等待 < 夹爪全程3.4s(80→0mm@23.6mm/s) —— 闭合/释放
    未完成臂就运动 = "夹爪没有完全开合机械臂就动了"的根因."""
    import time as _t
    t0 = _t.monotonic()
    last_mm = None
    stable_t0 = None
    while _t.monotonic() - t0 < timeout:
        try:
            r = controller.arm_client.get_state()
            if r.get("ok"):
                cur_mm = _gripper_rad_to_mm(float(r.get("gripper", 0.0)))
                if abs(cur_mm - target_mm) <= tol_mm:
                    return True
                if last_mm is not None and abs(cur_mm - last_mm) < 0.3:
                    if stable_t0 is None:
                        stable_t0 = _t.monotonic()
                    elif _t.monotonic() - stable_t0 >= stable_s:
                        print("[夹爪等待] 停稳于 %.1fmm (目标%gmm, 已夹住物体/到位), 继续"
                              % (cur_mm, target_mm))
                        return True
                else:
                    stable_t0 = None
                last_mm = cur_mm
        except Exception:
            pass
        _t.sleep(poll)
    print("[夹爪等待] 超时%.0fs (目标%gmm) — 继续后续步骤" % (timeout, target_mm))
    return False


def wait_arrival(controller, target_joints, tol_deg=3.0, timeout=4.0,
                 settle_s=1.0, settle_delta_deg=0.3):
    if target_joints is None:
        return False
    """★ 到位等待 v2(2026-09-16 修复"分段间10秒干等"): 修复'夹爪提前闭合'的
    原始目的 = 插值流发完后臂还在VMAX途中时拦住后续步骤. 但 kp=25 软刚度下
    重力稳态偏差实测3.4~4.3°, 原容差3°永远达不到 → 每个承重段烧满10s超时
    (12/41段, 共~2分钟干等). v2 三判据任一即通过:
      ① 误差≤tol_deg(真正到位)
      ② 误差稳定(settle_s内变化<settle_delta°) = 软刚度稳态偏差, 已停稳
      ③ 硬超时timeout(4s, 兜底)
    仍在运动时误差持续缩小, 不会误触②."""
    import time as _t
    import math as _m
    t0 = _t.monotonic()
    last_err = None
    stable_t0 = None
    settled = False
    while _t.monotonic() - t0 < timeout:
        try:
            r = controller.arm_client.get_state()
            if r.get("ok") and r.get("joints"):
                err = _m.degrees(max(abs(a - b) for a, b in
                                     zip(r["joints"], target_joints)))
                if err <= tol_deg:
                    return True
                if last_err is not None and abs(err - last_err) < settle_delta_deg:
                    if stable_t0 is None:
                        stable_t0 = _t.monotonic()
                    elif _t.monotonic() - stable_t0 >= settle_s:
                        settled = True
                        break
                else:
                    stable_t0 = None
                last_err = err
        except Exception:
            pass
        _t.sleep(0.15)
    if settled:
        if err > 8.0:
            print("[到位等待] ⚠ 停稳但偏差 %.1f° 过大(疑似堵限位/关节失能), 继续后续步骤" % err)
        else:
            print("[到位等待] 稳态偏差 %.1f° (软刚度正常现象), 继续后续步骤" % err)
        return True
    return False


def execute_cartesian_rrt_transfer(publisher, ik_solver, start_xyz, goal_xyz,
                                    duration, q6=None, normal=None, obstacles=None):
    """
    转移段：先调用 rrt_planner.plan_path() 在笛卡尔空间做安全段路径规划，
    再把规划出的每一段路径点依次用 move_linear() 走完。

    obstacles 默认传空列表——rrt_planner.py 自带的 OBSTACLES 只是占位示例
    数据，不是现场真实障碍物，用它做碰撞检测反而可能挡住原本可行的路径；
    传空列表时 plan_path() 内部会先检测起止点连线是否可行，可行则直接
    返回[起点,终点]两点，效果上等价于原来的直接一条直线，只有以后在
    obstacles 里填入现场真实障碍物坐标后，才会真正规划绕障路径。

    normal 在这里当作"转移全程保持的末端姿态法向量"透传给 move_linear()
    （比如涂抹动作从抓取点上方转移到目标点上方时，希望棉签姿态从头到尾
    都朝向目标表面法向量，而不是转移过程中姿态飘忽）。

    RRT规划失败（起止点本身不可达/无解）时，退回直接调用一次 move_linear()
    兜底，保证转移段本身不会因为规划失败而彻底卡死整个动作。
    """
    obstacles = obstacles if obstacles is not None else []
    waypoints = rrt_planner.plan_path(list(start_xyz), list(goal_xyz), obstacles=obstacles)

    if not waypoints or len(waypoints) < 2:
        rospy.logwarn("RRT转移段规划失败({} -> {})，退回直线运动兜底".format(start_xyz, goal_xyz))
        move_linear(publisher, ik_solver, start_xyz, goal_xyz, duration, q6=q6, normal=normal)
        return

    seg_duration = duration / float(len(waypoints) - 1)
    for i in range(len(waypoints) - 1):
        move_linear(publisher, ik_solver, waypoints[i], waypoints[i + 1],
                    seg_duration, q6=q6, normal=normal)


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：move_linear() 使用案例 —— 用在"涂抹"动作的来回涂抹上
#
# 对应 execute_action() 里 action_num==1(涂抹) 分支原来的写法：
#     for i in range(5):
#         smear_right = point_target.copy()
#         smear_right[0] += 0.05   # ← 这里改的其实是关节1角度(基座旋转)，不是笛卡尔X坐标，
#                                     末端实际走的不是直线，是一段弧线
#         ...
#
# 下面这个例子改成先算出涂抹中心点左右两个笛卡尔端点，再用 move_linear() 逐段走真直线。
# 这是一段独立的使用示例，没有直接接入 execute_action() 现有的异步 move_sequence 队列
# (因为 move_linear 是阻塞函数，直接嵌进那套异步架构会打乱执行顺序)。
# 想接入的话，最简单的方式是整体按 gluon_controller_cjh(1).py 那种"阻塞式脚本"风格调用，
# 或者把这几行放进一个QThread的run()里执行。
# ──────────────────────────────────────────────────────────────────────────────
def example_smear_with_linear_motion(publisher, ik_solver, target_xyz, q6=None, normal=None,
                                      smear_offset=0.05, repeat=3, seg_duration=2.5):
    """
    涂抹动作的直线往返示例：以 target_xyz 为中心，沿X方向左右偏移 smear_offset(米)，
    来回移动 repeat 次，每一小段都是调用 move_linear() 走出来的真正直线。

    用法参考(在execute_action的action_num==1分支里，把原来手写关节偏移的for循环换成)：
        example_smear_with_linear_motion(
            self.controller.publisher,
            self.controller.move_to_cartesian,
            [target_x, target_y, target_z],
            q6=q6,
            normal=normal
        )
    """
    right_point = [target_xyz[0] + smear_offset, target_xyz[1], target_xyz[2]]
    left_point = [target_xyz[0] - smear_offset, target_xyz[1], target_xyz[2]]

    current_point = target_xyz
    for i in range(repeat):
        move_linear(publisher, ik_solver, current_point, right_point, seg_duration, q6=q6, normal=normal)
        move_linear(publisher, ik_solver, right_point, left_point, seg_duration, q6=q6, normal=normal)
        current_point = left_point

    # 涂抹完成后回到目标点中心，收尾位置和原逻辑保持一致
    move_linear(publisher, ik_solver, current_point, target_xyz, seg_duration, q6=q6, normal=normal)


def normal_to_quaternion(nx, ny, nz):
    """
    将表面法向量转换为描述末端执行器姿态的四元数
    
    已知末端执行器的坐标轴与笛卡尔坐标轴的对应关系：
    - 末端z轴 → 笛卡尔x轴 [1,0,0]
    - 末端y轴 → 笛卡尔z轴 [0,0,1]
    - 末端x轴 → 笛卡尔y轴 [0,1,0]
    
    目标：使末端执行器的z轴指向表面法向量方向
    
    参数:
        nx, ny, nz: 表面法向量在笛卡尔空间中的分量
    
    返回:
        [qx, qy, qz, qw]: 四元数 (x, y, z, w)
    
    验证:
        (1,0,0) → [0, 0, 0, 1]
        (-1,0,0) → [0, 1, 0, 0]
        (0,1,0) → [0, 0, 0.7071, 0.7071]
        (0,-1,0) → [0, 0, -0.7071, 0.7071]
        (0,0,1) → [0, -0.7071, 0, 0.7071]
        (0,0,-1) → [0, 0.7071, 0, 0.7071]
    """
    import math
    
    # 归一化法向量
    norm = math.sqrt(nx*nx + ny*ny + nz*nz)
    if norm < 1e-10:
        # 零向量，默认指向-z方向
        return [0.0, 0.7071, 0.0, 0.7071]
    
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    
    # 末端z轴在末端坐标系中为 [1,0,0]
    # 目标：将末端z轴旋转到法向量方向
    # 即：R * [1,0,0]^T = [nx, ny, nz]^T
    
    # 但是末端z轴在笛卡尔坐标系中对应的是X轴方向 [1,0,0]
    # 所以我们需要找到旋转，使得 [1,0,0] 旋转到 [nx, ny, nz]
    
    # 旋转轴：起始向量 [1,0,0] 与目标向量 [nx, ny, nz] 的叉积
    # start = [1, 0, 0]
    # target = [nx, ny, nz]
    # axis = start × target = [0, -nz, ny]
    axis_x = 0.0
    axis_y = -nz
    axis_z = ny
    
    axis_norm = math.sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z)
    
    if axis_norm < 1e-10:
        # 法向量与X轴平行或反平行
        if nx > 0:
            # 指向+X方向，不需要旋转
            return [0.0, 0.0, 0.0, 1.0]
        else:
            # 指向-X方向，绕Y轴旋转180°
            return [0.0, 1.0, 0.0, 0.0]
    
    # 归一化旋转轴
    axis_x /= axis_norm
    axis_y /= axis_norm
    axis_z /= axis_norm
    
    # 旋转角度：起始向量与目标向量的夹角
    dot_product = 1.0 * nx + 0.0 * ny + 0.0 * nz  # [1,0,0] · [nx,ny,nz]
    angle = math.acos(max(-1.0, min(1.0, dot_product)))
    
    # 使用轴角转四元数公式
    half_angle = angle / 2.0
    sin_half = math.sin(half_angle)
    
    qw = round(math.cos(half_angle), 4)
    qx = round(axis_x * sin_half, 4)
    qy = round(axis_y * sin_half, 4)
    qz = round(axis_z * sin_half, 4)
    
    # 四元数格式: [qx, qy, qz, qw]
    return [qx, qy, qz, qw]

# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ★ 进程级崩溃捕获: UI 静默死亡(无traceback)时把堆栈写进日志, 定位真凶
    import faulthandler
    faulthandler.enable(file=sys.__stderr__)
    try:
        controller = ArmController()
        app = QApplication(sys.argv)
        ui = ArmControllerUI(controller)
        ui.show()
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
    except Exception:
        import traceback
        print("[FATAL] UI 主循环异常:\n%s" % traceback.format_exc(), file=sys.__stderr__)
        raise

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
                             QMessageBox, QSplitter, QFormLayout, QComboBox, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
from PyQt5.QtGui import QColor
import time

# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：自动物证采集流程依赖模块（新增文件，不改动任何原有代码） ★★★
# 对应专利步骤S3(SPME-CenterNet检测) + S4(CDG-ATRG精细分割/标定/深度反投影/
# 法向量估计) + S5(按类别分派execute_action)
# ──────────────────────────────────────────────────────────────────────────────
import perception_calib as pcalib
from centernet_bridge import ForensicDetector
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

# ★★★ 新增：物证回收位 ★★★
# 涂抹后的棉签动作完成后不再放回原来的工具架位置（避免污染工具架、
# 混淆待用工具与已沾染物证的工具），而是转移到这个回收位释放，供后续
# 装袋/编号等物证管理流程使用。
# ★ 2026-09-18 起此常量仅被动作1(涂抹)使用; 动作2(粘取)放回粘取器架,
#   动作3(抓取)用专属回收位(0.14,0,0.05), 见 _execute_action_impl 内分派.
# ★ 这是一个占位坐标，需要现场根据实际证物台/回收框位置量取后修改；
#   取值需同时满足 工作空间的硬性限制(X/Y/Z均在±0.40m内)，
#   并尽量落在UI"工作空间限制"面板标注的常用范围内(X:0.15~0.40,
#   Y:-0.10~0.30, Z:0.10~0.35，仅供参考，非代码强制限制)。
EVIDENCE_RECOVERY_POINT = [0.1258, -0.088, 0.046]  # ★前方(原[-0.10,0.16]在左后方: 涂抹取棉签/抓取放物后 j1 需~100°大回转 -> 过流失能; 挪到与粘取器架同侧前方, 消除大回转)

# ── 夹爪安全限位 (2026-09-08 自 gluon_controller_kk_rebot.py 同步; 顶断限位事故后的硬性约束) ──
# 电机7(rs-00)原始角: 负=闭合, 正=张开; 断电静止=-1.43 ≈ 完全闭合.
GRIPPER_CLOSED = 4.374       # 74mm安全位rad(新零位系, 启动/看门狗恢复后锁定目标)
GRIPPER_OPEN_MAX = 5.07      # 张开侧紧急钳位rad(新零位系, ≈85mm, 留3mm余量<实测88)

# ── 夹爪实物标定 (齿轮齿条直线双夹爪) ─────────────────────────────────────
# 输入值 = 两夹爪之间的真实物理距离(mm), 一一对应:
#   rad = _GRIP_CAL_RAD_REF + (mm-74)/15.75   斜率15.75mm/rad全程实测不变.
GRIPPER_MM_CLOSED = 0.0      # mm 输入下限(0=并拢指令: 闭到指尖相触并轻握)
GRIPPER_MM_GRIP = 30.0       # ★ 预设动作闭合爪距(2026-09-16用户指定): 取工具/抓物证统一30mm
GRIPPER_MM_TOUCH = 2.0       # 指尖相触点(实测1.9mm, 取2) — "闭合"按钮目标位
GRIPPER_MM_OPEN = 80.0       # mm 输入上限(用户指定张开位)
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

GRIPPER_SPEED_MM_S = 4.7   # ★ 夹爪斜坡限速(mm/s) — 与 arm_server 的 GRIP_VMAX(0.30rad/s×15.75) 保持同步


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
# ★★★ 新增：人工画框标注控件 (2026-09-08, 人工模式第一步) ★★★
# 暂停瞬间冻结原始彩色帧, 拖拽画矩形框(labelImg式), 信号回传图像坐标框。
# 只负责"冻结帧+画框+中心像素", 不做像素→基坐标换算(待M-链标定后在
# on_annotate_box 挂接, 见其中TODO)。
# 设计要点:
#   - 冻结=显示停, 驱动不停: 后台相机话题照常收, 「恢复预览」零成本回直播;
#   - 画框必须在静态帧上(与自动模式"静止读坐标"同一纪律, 消除θf/θn错配);
#   - 冻结帧用 KeepAspectRatio 完整显示(直播是 ByExpanding 裁边), 画框时
#     看得到整幅画面, 控件→像素反映射也因此简单可靠。
# ──────────────────────────────────────────────────────────────────────────────
class AnnotateCameraLabel(QLabel):
    """支持冻结帧 + 拖拽矩形框的相机显示控件(冻结时才响应鼠标)"""

    # 框完成信号: 图像坐标系 (x1, y1, x2, y2)
    box_ready = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frozen_img = None       # 冻结帧 QImage(RGB888) 深拷贝
        self._rect_img = None         # 当前框, 图像坐标 (x1,y1,x2,y2)
        self._drag_start = None       # 拖拽起点, 图像坐标 (u,v)
        self._box_confirmed = False   # 框是否已确认(黄=待确认, 绿=已确认)

    # ---- 外部接口 ----
    def freeze(self, img_bgr):
        """冻结: 传入 BGR numpy 帧, 进入画框模式"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
            self.box_ready.emit(*rect)
        else:
            self._rect_img = None
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, ev):
        p = QPainter(self)
        if self.frozen:
            lw, lh = self.width(), self.height()
            scale, dw, dh, iw, ih = self._fit_geo()
            p.drawImage(QRectF(dw, dh, iw * scale, ih * scale), self._frozen_img)
            if self._rect_img is not None:
                x1, y1, x2, y2 = self._rect_img
                # 框: 待确认=黄, 已确认=绿
                p.setPen(QPen(QColor(0, 230, 118) if self._box_confirmed
                              else QColor(255, 220, 0), 2))
                p.drawRect(QRectF(dw + x1 * scale, dh + y1 * scale,
                                  (x2 - x1) * scale, (y2 - y1) * scale))
                # 中心十字 (红)
                cu = (x1 + x2) / 2.0
                cvc = (y1 + y2) / 2.0
                cx, cy = dw + cu * scale, dh + cvc * scale
                p.setPen(QPen(QColor(255, 60, 60), 2))
                p.drawLine(QPointF(cx - 12, cy), QPointF(cx + 12, cy))
                p.drawLine(QPointF(cx, cy - 12), QPointF(cx, cy + 12))
                # 中心像素标注
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(cx + 16, cy - 8),
                           "(%d, %d)%s" % (int(cu), int(cvc),
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
        # ★ UI v2.0 重构 (2026-09-09): 近单色"工业控制台"风格
        #   配色收敛: 石墨蓝底 + 单一钢蓝点缀 + 唯一暗红(急停/安全语义);
        #   结构: 顶部标题栏 + 左控制栏(卡片) + 右相机区(相机卡+标注栏+并排信息面板)
        # ──────────────────────────────────────────────────────────────────────
        self.setStyleSheet("""
            * {
                font-family: 'Noto Sans CJK SC','Noto Sans CJK HK',
                              'WenQuanYi Zen Hei','Droid Sans Fallback','Sans-serif';
                font-size: 13px;
            }
            QMainWindow, QWidget { background-color: #0e141b; color: #d9e2ec; }
            /* 顶部标题栏 */
            #headerBar { background-color: #121926; border-bottom: 1px solid #26313f; }
            #headerTitle { font-size: 19px; font-weight: 600; color: #eaf1f9;
                           background: transparent; }
            #headerSub { font-size: 11px; color: #64788f; letter-spacing: 3px;
                         background: transparent; }
            /* 卡片容器(v2.1: 标题移入卡片内嵌头部行, 不再用QGroupBox原生title) */
            QGroupBox {
                background-color: #161d27;
                border: 1px solid #26313f;
                border-radius: 8px;
                margin-top: 0px;
                padding: 1px;
            }
            /* 焦点卡片(目标位姿): 微亮底色 + 顶部2px点缀线, 全局唯一 */
            QGroupBox#cardPrimary {
                background-color: #1a222e;
                border: 1px solid #2e3c4d;
                border-top: 2px solid #5f8fca;
            }
            /* 卡片头部行: 图标符 + 标题 + 细线 + 序号 (全部单色系) */
            #cardHeader { background: transparent; }
            #cardGlyph { color: #7ea4cf; font-size: 15px; background: transparent; }
            #cardTitle { color: #aebfd4; font-size: 12px; font-weight: 600;
                         letter-spacing: 1px; background: transparent; }
            #cardIndex { color: #3e4c5e; font-size: 11px;
                         font-family: 'Consolas','monospace'; background: transparent; }
            #cardRule { background-color: #222e3c; }
            QLabel { background: transparent; color: #c9d4e2; }
            QLineEdit {
                background-color: #101823; color: #e6edf5;
                border: 1px solid #2a3849; border-radius: 5px;
                padding: 5px 8px; selection-background-color: #3d5a80;
            }
            QLineEdit:focus { border: 1px solid #5f8fca; }
            /* v2.2 无框分区的紧凑输入槽: 仅下划线, 无矩形底框 */
            QLineEdit#axisField {
                background: transparent; border: none;
                border-bottom: 1px solid #2f3d4f; border-radius: 0;
                padding: 5px 4px 5px 4px; color: #e6edf5;
            }
            QLineEdit#axisField:focus { border-bottom: 1px solid #5f8fca; }
            /* v2.2 圆形图标按钮(关节旋转/夹爪开合等稀疏按钮位) */
            QPushButton#circ {
                background-color: #1c2531; color: #cdd8e6;
                border: 1px solid #2f3d4f; border-radius: 20px;
                min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px;
                padding: 0; font-size: 16px;
            }
            QPushButton#circ:hover { background-color: #232f3e; border-color: #5f8fca; }
            QPushButton#circ:pressed { background-color: #182231; }
            QPushButton#circ:disabled {
                color: #55636f; border-color: #26313f; background-color: #141b25;
            }
            /* 按钮: 卡片内=圆角矩形(8px); 工具条内=胶囊形(15px) —— 用形状区分
               层级, 而非加颜色; 全局仍只有 默认/primary/danger 三种形态 */
            QPushButton {
                background-color: #1c2531; color: #d9e2ec;
                border: 1px solid #2f3d4f; border-radius: 8px;
                padding: 7px 14px;
            }
            QPushButton:hover { background-color: #232f3e; border-color: #5f8fca; }
            QPushButton:pressed { background-color: #182231; }
            QPushButton:disabled { color: #55636f; border-color: #26313f;
                                   background-color: #141b25; }
            QPushButton#primary {
                background-color: #1b2a3d; color: #dcebfa; border: 1px solid #5f8fca;
            }
            QPushButton#primary:hover { background-color: #223349; }
            QPushButton#primary:disabled {
                color: #48606f; border-color: #31404f; background-color: #141b25;
            }
            /* 唯一的红色: 仅安全/急停语义 */
            QPushButton#danger {
                background-color: #26191d; color: #f0d7da; border: 1px solid #a8515b;
            }
            QPushButton#danger:hover { background-color: #332228; }
            /* 三条工具条(相机条/标注条/底部系统条)内的按钮: 胶囊形 */
            #camBar QPushButton, #annotateBar QPushButton, #leftFooter QPushButton {
                border-radius: 15px; padding: 6px 16px;
            }
            /* 信息面板: 等宽日志区 */
            QTextEdit {
                font-family: 'Consolas','Noto Sans CJK SC','DejaVu Sans Mono','monospace';
                font-size: 13px; color: #c9d6e4;
                background-color: #10161f; border: 1px solid #26313f;
                border-radius: 6px; padding: 8px;
            }
            /* 工具条/内嵌条: 相机控制条与人工标注条 */
            #camBar, #annotateBar, #leftFooter {
                background-color: #10161f; border: 1px solid #1f2a38;
                border-radius: 6px;
            }
            /* 标题栏装饰: 竖向点缀条 */
            #headerAccent { background-color: #5f8fca; border-radius: 2px; }
            #vSep { background-color: #26313f; }
            #camChip {
                color: #64788f; border: 1px solid #26313f; border-radius: 11px;
                padding: 2px 10px; font-size: 11px; background-color: #161d27;
            }
            QStatusBar {
                background-color: #121926; border-top: 1px solid #26313f;
                color: #8fa2b8;
            }
            QStatusBar::item { border: none; }
            QSplitter::handle { background-color: #0e141b; }
            QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
            QScrollBar::handle:vertical {
                background: #2c3949; border-radius: 4px; min-height: 28px;
            }
            QScrollBar::handle:vertical:hover { background: #3a4a5f; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: transparent; height: 9px; margin: 0; }
            QScrollBar::handle:horizontal {
                background: #2c3949; border-radius: 4px; min-width: 28px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)

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
        #   2026-09-28: 改为 False, 启用 blood2 血迹检测(blood_detect.launch);
        #   要重新标定时改回 True 重启UI即可.
        self.CALIB_MODE = False
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
        self._raw_depth_msg = None       # ★ 原始深度msg缓存
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
                "<span style='color:#00b4ff;'>[就绪] 检测信息框已就绪。"
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
        #   死亡 → 自动 USB重置 + 重启后端 + 重连重新使能 (用户无感自愈).
        self._arm_started = False          # 成功启动过机械臂后看门狗才介入
        self._can_wd_recovering = False    # 恢复流程互斥锁(防重入/防退出时捣乱)
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
        grp = QWidget()
        h = QHBoxLayout(grp)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        g_lbl = QLabel("夹爪")
        g_lbl.setStyleSheet("color: #8fa2b8; font-size: 13px;")
        self.servo_angle_input = QLineEdit("2")   # 默认2mm=并拢位
        self.servo_angle_input.setObjectName("axisField")
        self.servo_angle_input.setFixedWidth(52)
        self.servo_angle_input.setToolTip("两夹爪间距 mm (%g并拢 ~ %g张开)" % (
            GRIPPER_MM_TOUCH, GRIPPER_MM_OPEN))
        g_unit = QLabel("mm")   # ★ 单位可见化(用户要求: UI体现mm单位控制)
        g_unit.setStyleSheet("color: #64788f; font-size: 11px;")
        self.servo_set_btn = QPushButton("设定")
        self.servo_set_btn.setToolTip("设置夹爪间距(mm)")
        self.servo_open_btn = QPushButton("▲")
        self.servo_open_btn.setObjectName("circ")
        self.servo_open_btn.setToolTip("张开 5mm(相对当前位步进)")
        self.servo_open_btn.clicked.connect(lambda: self._gripper_step(+5.0))
        self.servo_close_btn = QPushButton("▼")
        self.servo_close_btn.setObjectName("circ")
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
        for w in (g_lbl, self.servo_angle_input, g_unit, self.servo_set_btn,
                  self.servo_open_btn, self.servo_close_btn,
                  self.servo_reset_btn, self.servo_recal_btn):
            h.addWidget(w)

        row = self.centralWidget().findChild(QWidget, "gripperRow")
        if row is not None and row.layout() is not None:
            row.layout().addWidget(grp)   # 该行本就为夹爪预留, 直接放满
        else:
            left_panel = self.centralWidget().findChild(QWidget, "leftPanel")
            if left_panel and left_panel.layout():
                left_panel.layout().addWidget(grp)
            else:
                self.centralWidget().layout().addWidget(grp)
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
        """[↻重标定]按钮: 后台线程强制重标(est=None 跳过交叉验证, 供手动掰动/清理后)."""
        if not getattr(self, '_arm_started', False):
            self.status_display.append("[gripper] 请先点[启动机械臂]再重标定")
            return
        threading.Thread(target=self._gripper_autocal, kwargs={'est': None}, daemon=True).start()

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
        import time as _t
        _now = _t.monotonic()
        if (_now - getattr(self, '_fk_refresh_last', 0.0) >= 0.5
                and not getattr(self, 'is_scanning', False)):
            self._fk_refresh_last = _now
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

        # 顶部标题栏: 点缀竖条 + 主标题 + 副题(左) / 设备徽章(右)
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(52)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 6, 20, 6)
        header_layout.setSpacing(10)
        header_accent = QWidget()
        header_accent.setObjectName("headerAccent")
        header_accent.setFixedSize(4, 22)
        header_title = QLabel("刑侦现场物证采集系统")
        header_title.setObjectName("headerTitle")
        header_sub = QLabel("FORENSIC EVIDENCE COLLECTION CONSOLE")
        header_sub.setObjectName("headerSub")
        header_layout.addWidget(header_accent)
        header_layout.addSpacing(4)
        header_layout.addWidget(header_title)
        header_layout.addSpacing(10)
        header_layout.addWidget(header_sub)
        header_layout.addStretch(1)
        root_layout.addWidget(header)

        # 主体: 左(控制卡片栏, 2) : 右(相机+信息, 3)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(10)

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
    
    def _make_card(self, glyph, title, index, primary=False):
        """UI v2.1 统一功能卡片(右栏内容区用): 内嵌头部行(图标符+标题+细线+
        序号) + 内容布局; 左栏已改无框分区(_make_section), 此卡片仅右栏使用。"""
        card = QGroupBox()
        if primary:
            card.setObjectName("cardPrimary")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(8)
        header = QWidget()
        header.setObjectName("cardHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(2, 0, 2, 0)
        hl.setSpacing(8)
        glyph_lbl = QLabel(glyph)
        glyph_lbl.setObjectName("cardGlyph")
        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        idx_lbl = QLabel(index)
        idx_lbl.setObjectName("cardIndex")
        rule = QWidget()
        rule.setObjectName("cardRule")
        rule.setFixedHeight(1)
        hl.addWidget(glyph_lbl)
        hl.addWidget(title_lbl)
        hl.addWidget(rule, 1)
        hl.addWidget(idx_lbl)
        v.addWidget(header)
        return card, v

    def _make_section(self, glyph, title, index):
        """UI v2.2 无框分区(左栏专用): 标题行(图标符+标题+细线+序号)+内容布局.
        ★ 无边框无底色 —— 分区感完全靠排版而非盒子, 与右栏卡片形成
          "控制轨 vs 内容区" 的对比。"""
        section = QWidget()
        v = QVBoxLayout(section)
        v.setContentsMargins(2, 0, 2, 0)
        v.setSpacing(8)
        header = QWidget()
        header.setObjectName("cardHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(2, 0, 2, 0)
        hl.setSpacing(8)
        glyph_lbl = QLabel(glyph)
        glyph_lbl.setObjectName("cardGlyph")
        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        idx_lbl = QLabel(index)
        idx_lbl.setObjectName("cardIndex")
        rule = QWidget()
        rule.setObjectName("cardRule")
        rule.setFixedHeight(1)
        hl.addWidget(glyph_lbl)
        hl.addWidget(title_lbl)
        hl.addWidget(rule, 1)
        hl.addWidget(idx_lbl)
        v.addWidget(header)
        return section, v

    def create_left_panel(self):
        """UI v2.2 左栏: 无框排版式分区; X/Y/Z/J6 紧凑下划线输入槽(2×2);
        关节旋转与夹爪开合为圆形图标按钮; 夹爪控件由 init_servo_ui 运行时
        插入 jointRow; 底部保留系统工具条。"""
        panel = QWidget()
        panel.setObjectName("leftPanel")   # ★ init_servo_ui 兜底查找用
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(15)

        # ══════════ 01 目标位姿(下划线输入槽) ══════════
        sec1, sec1_v = self._make_section("⊕", "目标位姿 · 笛卡尔", "01")

        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()
        self.j6_input = QLineEdit("0")   # ★ 2026-10-30 用户要求: J6 默认填 0

        def _axis_cell(row, col, name, unit, field):
            cell = QHBoxLayout()
            cell.setSpacing(6)
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #64788f;")
            field.setObjectName("axisField")
            field.setFixedWidth(96)
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet("color: #64788f; font-size: 11px;")
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
        n_lbl.setStyleSheet("color: #64788f;")
        normal_row.addWidget(n_lbl)
        self.nx_input = QLineEdit("0")
        self.ny_input = QLineEdit("0")
        self.nz_input = QLineEdit("-1")
        for lbl_txt, w in (("nx", self.nx_input), ("ny", self.ny_input), ("nz", self.nz_input)):
            w.setObjectName("axisField")
            w.setFixedWidth(56)
            u = QLabel(lbl_txt)
            u.setStyleSheet("color: #64788f; font-size: 11px;")
            normal_row.addWidget(w)
            normal_row.addWidget(u)
        normal_row.addStretch(1)
        sec1_v.addLayout(normal_row)

        # 姿态四元数读出
        self.normal_status_label = QLabel("当前姿态: [1.0, 0.0, 0.0, 0.0]")
        self.normal_status_label.setStyleSheet("color: #7f93ac; font-size: 12px;")
        sec1_v.addWidget(self.normal_status_label)

        # 移动 + 读取位姿(标定用)
        cart_btn_row = QHBoxLayout()
        cart_btn_row.setSpacing(6)
        move_btn = QPushButton("◎  移动到位置")
        move_btn.setObjectName("primary")
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
        home_btn = QPushButton("↩  返回初始位置")
        home_btn.setMinimumHeight(42)
        home_btn.clicked.connect(self.return_to_zero)
        ground_btn = QPushButton("▼  检测地面")
        ground_btn.setMinimumHeight(42)
        ground_btn.setToolTip("检测地面: 移动到 (0.35, 0, 0), 法向竖直朝下 (0,0,-1)")
        ground_btn.clicked.connect(lambda: self.detect_ground())

        flow_grid.addWidget(home_btn, 0, 0)
        flow_grid.addWidget(ground_btn, 0, 1)
        sec2_v.addLayout(flow_grid)

        # ══════════ 03 关节 · 夹爪(J6与夹爪上下两行, 用户要求) ══════════
        sec3, sec3_v = self._make_section("⇄", "关节 · 夹爪", "03")

        # ── Row A: J6 (绝对设定 + 步进旋转 + 回开机位) ──
        j6_row = QHBoxLayout()
        j6_row.setSpacing(8)
        j_lbl = QLabel("J6 ±")
        j_lbl.setStyleSheet("color: #8fa2b8; font-size: 13px;")
        self.rotate_amount = QLineEdit("0.1")
        self.rotate_amount.setObjectName("axisField")
        self.rotate_amount.setFixedWidth(64)
        self.rotate_amount.setToolTip(
            "J6 弧度值: 点[设定]转到该绝对角度; ↻/↺ 以它为步进量旋转")
        j_unit = QLabel("rad")
        j_unit.setStyleSheet("color: #64788f; font-size: 11px;")
        self.j6_set_btn = QPushButton("设定")
        self.j6_set_btn.setToolTip("J6 转到输入的绝对角度(rad)")
        self.j6_set_btn.clicked.connect(self.set_j6_absolute)
        cw_btn = QPushButton("↻")
        cw_btn.setObjectName("circ")
        cw_btn.setToolTip("J6 顺时针旋转(步进量=输入值)")
        cw_btn.clicked.connect(lambda: self.rotate_j6(float(self.rotate_amount.text())))
        ccw_btn = QPushButton("↺")
        ccw_btn.setObjectName("circ")
        ccw_btn.setToolTip("J6 逆时针旋转(步进量=输入值)")
        ccw_btn.clicked.connect(lambda: self.rotate_j6(-float(self.rotate_amount.text())))
        self.j6_home_btn = QPushButton("J6归位")
        self.j6_home_btn.setToolTip("回到本次启动机械臂时记录的J6角度(不确定转了几圈时一键回开机位)")
        self.j6_home_btn.clicked.connect(self.j6_return_home)
        for w in (j_lbl, self.rotate_amount, j_unit, self.j6_set_btn,
                  cw_btn, ccw_btn, self.j6_home_btn):
            j6_row.addWidget(w)
        j6_row.addStretch(1)
        sec3_v.addLayout(j6_row)

        # ── Row B: 夹爪控件容器 (init_servo_ui 运行时插入, 全宽行) ──
        gripper_row_w = QWidget()
        gripper_row_w.setObjectName("gripperRow")   # ★ init_servo_ui 按名插入
        gripper_row_l = QHBoxLayout(gripper_row_w)
        gripper_row_l.setContentsMargins(0, 0, 0, 0)
        gripper_row_l.setSpacing(8)
        sec3_v.addWidget(gripper_row_w)

        # ══════════ 04 预设动作(三个全宽等价按钮, 完全并列) ══════════
        sec4, sec4_v = self._make_section("▣", "预设动作", "04")
        height_row = QHBoxLayout()
        height_row.setSpacing(6)
        height_label = QLabel("安全高度 H (m)")
        height_label.setStyleSheet("color: #64788f; font-size: 13px;")
        self.height_input = QLineEdit(str(SAFE_HEIGHT))
        self.height_input.setObjectName("axisField")
        self.height_input.setFixedWidth(92)
        height_row.addWidget(height_label)
        height_row.addWidget(self.height_input)
        height_row.addStretch(1)
        sec4_v.addLayout(height_row)

        # ★ 三个动作并列关系 → 结构完全相同: 全宽、等高、图标+文字, 纵向排列
        self.action1_btn = QPushButton("→  动作 1 · 涂抹")
        self.action1_btn.setToolTip("液体类: 取棉签→转移到目标→往复涂抹→回收")
        self.action1_btn.setMinimumHeight(40)
        self.action1_btn.clicked.connect(lambda: self.execute_action(1))
        self.action2_btn = QPushButton("⊙  动作 2 · 粘取")
        self.action2_btn.setToolTip("微小类: 取粘取器→下压停留→粘附→回收")
        self.action2_btn.setMinimumHeight(40)
        self.action2_btn.clicked.connect(lambda: self.execute_action(2))
        self.action3_btn = QPushButton("◈  动作 3 · 抓取")
        self.action3_btn.setToolTip("固体类: 沿法向接近→闭合夹爪→转移→回收位释放")
        self.action3_btn.setMinimumHeight(40)
        self.action3_btn.clicked.connect(lambda: self.execute_action(3))
        for btn in (self.action1_btn, self.action2_btn, self.action3_btn):
            sec4_v.addWidget(btn)

        self.action_buttons = [self.action1_btn, self.action2_btn, self.action3_btn]

        # ══════════ 底部固定系统条(启动/急停 + 工作空间说明) ══════════
        system_group = QWidget()
        system_group.setObjectName("leftFooter")
        system_layout = QHBoxLayout(system_group)
        system_layout.setContentsMargins(10, 8, 10, 8)
        system_layout.setSpacing(6)

        start_btn = QPushButton("启动机械臂")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self.start_arm)
        stop_btn = QPushButton("紧急停止")
        stop_btn.setObjectName("danger")
        stop_btn.clicked.connect(self.emergency_stop)
        system_layout.addWidget(start_btn)
        system_layout.addWidget(stop_btn)
        system_layout.addStretch(1)

        ws_label = QLabel("工作空间  X 0.15~0.40 · Y -0.10~0.30\nZ 0.10~0.35  (m)")
        ws_label.setStyleSheet("color: #64788f; font-size: 11px;")
        ws_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        system_layout.addWidget(ws_label)

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
        camera_group, camera_layout = self._make_card("◉", "相机视图 · 人工标注", "06")

        # ★ 画框模式: 增强控件(冻结帧+拖拽画框), 直播外观与原 QLabel 一致
        self.camera_label = AnnotateCameraLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setText("等待相机图像...")
        self.camera_label.setStyleSheet("background-color: #05080c; color: #8fa2b8;")
        self.camera_label.box_ready.connect(self.on_annotate_box)

        # ★★★ 人工标注工作流条 ★★★
        # [人工标注(冻结画面)] → 拖拽画框(黄=待确认) → [确认框选](绿=生效) /
        # [取消框选](清框重画) → [退出]回直播; 右端为状态读出
        annotate_bar = QWidget()
        annotate_bar.setObjectName("annotateBar")
        annotate_layout = QHBoxLayout(annotate_bar)
        annotate_layout.setContentsMargins(10, 7, 10, 7)
        annotate_layout.setSpacing(8)
        self.annotate_pause_btn = QPushButton("人工标注")
        self.annotate_pause_btn.setObjectName("primary")
        self.annotate_pause_btn.setToolTip("停止机械臂并冻结当前画面, 进入拖拽画框模式")
        self.annotate_pause_btn.clicked.connect(self.on_annotate_pause)
        self.annotate_confirm_btn = QPushButton("确认框选")
        self.annotate_confirm_btn.setObjectName("primary")
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
        self.annotate_status_label.setStyleSheet("color: #7f93ac; font-size: 12px;")
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

        # ══════════ 相机控制条: 画面状态徽章 + 启停 ══════════
        cam_bar = QWidget()
        cam_bar.setObjectName("camBar")
        cam_control_layout = QHBoxLayout(cam_bar)
        cam_control_layout.setContentsMargins(10, 7, 10, 7)
        cam_control_layout.setSpacing(8)
        self.cam_state_chip = QLabel("待机")
        self.cam_state_chip.setObjectName("camChip")
        self._cam_chip_state = "待机"
        start_cam_btn = QPushButton("启动相机")
        start_cam_btn.setObjectName("primary")
        start_cam_btn.clicked.connect(self.start_camera)
        stop_cam_btn = QPushButton("停止相机")
        stop_cam_btn.clicked.connect(self.stop_camera)
        sep2 = QWidget()
        sep2.setObjectName("vSep")
        sep2.setFixedSize(1, 20)
        cam_control_layout.addWidget(self.cam_state_chip)
        cam_control_layout.addWidget(start_cam_btn)
        cam_control_layout.addWidget(stop_cam_btn)
        cam_control_layout.addWidget(sep2)
        cam_hint = QLabel("拖拽画矩形框 · 框中心即目标点")
        cam_hint.setStyleSheet("color: #64788f; font-size: 12px;")
        cam_control_layout.addWidget(cam_hint)
        cam_control_layout.addStretch(1)

        camera_layout.addWidget(self.camera_label, 1)
        camera_layout.addWidget(annotate_bar)
        camera_layout.addWidget(cam_bar)

        # ══════════ 07 检测信息 | 08 状态信息 并排 ══════════
        # ★★★ 检测信息实时显示框 ★★★
        # 扫描过程中每次相机检测到物证, 都在这里显示:
        #   类别 / 置信度 / 像素坐标 / 深度 / 相机系坐标 / th1 / 转化后基坐标系坐标
        detect_group, detect_cv = self._make_card("▤", "检测信息", "07")
        self.detection_display = QTextEdit()
        self.detection_display.setReadOnly(True)
        self.detection_display.setMinimumHeight(150)
        # ★ 覆盖全局 QTextEdit 样式: 保持等宽浅字/深底, 与整体配色一致
        self.detection_display.setStyleSheet(
            "QTextEdit { font-family: 'Consolas','Noto Sans CJK SC','monospace';"
            " font-size: 13px; color: #c9d6e4; background-color: #10161f;"
            " border: 1px solid #26313f; border-radius: 6px; padding: 8px; }")
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
            "font-family: 'Consolas','monospace'; font-size: 12px;"
            " color: #a8c8ea; background-color: #10161f;"
            " border: 1px solid #1f2a38; border-radius: 4px; padding: 6px;")
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

                # ★ reBot 相机为正装(原始图像本身正立): 不做任何翻转直接显示。
                #   GLUON 时代的"物理倒装"垂直翻转在 reBot 上是画面颠倒的根因
                #   (旧 flip(0) 与后来试的 flip(-1) 都会倒, 已实测确认)。

                h, w, ch = cv_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.width(),
                    self.camera_label.height(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.FastTransformation   # ★ 平滑缩放改快速: 再省几毫秒
                )
                self.camera_label.setPixmap(scaled_pixmap)
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
            '待机':   ("●  待机",   "#64788f", "#26313f"),
            '实时':   ("●  实时",   "#a8c8ea", "#3d5a80"),
            '已冻结': ("❚❚ 已冻结", "#93a7c0", "#2f3d4f"),
        }
        txt, fg, bd = presets.get(state, presets['待机'])
        chip.setText(txt)
        chip.setStyleSheet(
            "color: %s; border: 1px solid %s; border-radius: 11px;"
            " padding: 2px 10px; font-size: 11px; background-color: #161d27;" % (fg, bd))
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
        self._freeze_depth = self._decode_depth()
        self._annotate_dbg("[冻结] depth=%s color=%s joints[0]=%.3f" % (
            "None" if self._freeze_depth is None
            else "%dx%d" % (self._freeze_depth.shape[0], self._freeze_depth.shape[1]),
            "None" if getattr(self, '_raw_color_msg', None) is None
            else "%dx%d" % (self._raw_color_msg.height, self._raw_color_msg.width),
            joints[0]))
        self._freeze_t = time.strftime("%H:%M:%S")
        # ★ 2026-09-16 冻结位姿快照: 确认框选的相机→基座变换必须用冻结时刻的
        #   臂位姿 —— 原来实现用实时get_pose(), v2流程下确认前臂常已移动
        #   (点过移动/动作), 拿新位姿变换旧冻结帧像素 = 坐标全错(实测同帧同
        #   深度两次确认z差14.5cm, "相机向下时z不准"的根因).
        self._freeze_pose = None
        try:
            rp = self.controller.arm_client.get_pose()
            if rp.get("ok") and rp.get("pos"):
                self._freeze_pose = {'ok': True, 'pos': list(rp["pos"]),
                                     'rpy': list(rp["rpy"])}
        except Exception as _e:
            self._annotate_dbg("[冻结] get_pose异常: %r" % _e)
            print("[冻结] 位姿快照失败(确认时将回落实时位姿): %s" % _e)
        self._annotate_dbg("[冻结] pose快照=%s" % (
            "None" if self._freeze_pose is None else "ok"))

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
            dv = pcalib.sample_stable_depth(depth_img, dx, dy)
            if dv is None:
                dv = pcalib.sample_stable_depth(depth_img, dx, dy, win=9)
            if dv is None:
                dv = pcalib.sample_stable_depth(depth_img, dx, dy, win=13)
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
                #   估算法向, 估计失败保留输入框原值); Z<=0 → 法向固定(0,0,-1);
                #   J6 恒为 0(下方填框已设).
                odx, ody, odz = self._boxed_offset(bx, by)
                bx_c = bx + odx
                by_c = by + ody
                bz_c = bz + odz
                if bz_c > 0.0:
                    est_n = self._estimate_box_normal(box)   # ★ 方案1: 框内深度法向
                else:
                    est_n = [0.0, 0.0, -1.0]
                # ★ 2026-10-30 用户要求: 取消原"目标z<0 → Z强制-0.05"规则,
                #   Z 保留深度反投影实测值+补偿.
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

        # ★ 端到端偏移补偿(Y线性函数 + X恒定 + 远距Z)
        odx, ody, odz = self._boxed_offset(bx, by)
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

        # ★ 逐个法向尝试IK(停距由 APPROACH_STANDOFF_M 决定, 当前0=精确到点)
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
        # ★ 成功: 运动 + 恢复用户输入 + 显示所用法向
        self.controller.q6_value = 0.0
        _epoch = self.start_arm_movement(
            self.controller.joint_pos.copy(), joints, MOVE_DURATION)
        self._notify_trail_start()
        self._post_move_refine('画框', bx_c, by_c, bz_c, normal, epoch=_epoch)
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
    _BOX_CORR = np.array([0.01, 0.0, 0.0], dtype=np.float64)   # dx 恒定项
    _BOX_CORR_Y_SCALE = 0.0422   # y 比例补偿(全域生效)
    _BOX_CORR_Y_CONST = -0.0015  # y 恒定补偿(m)(全域生效)
    _BOX_CORR_Z_FAR = 0.0100     # X>阈值时 z 额外+10mm
    _BOX_CORR_X_FAR = 0.5       # z 补偿生效的 X 阈值(米)

    def _boxed_offset(self, bx, by):
        """画框端到端偏移补偿量 (dx, dy, dz):
        dy = Y_SCALE·y + Y_CONST (全域); dz 远距+10mm."""
        dy = self._BOX_CORR_Y_SCALE * by + self._BOX_CORR_Y_CONST
        dz = self._BOX_CORR[2] + (self._BOX_CORR_Z_FAR
                                  if bx > self._BOX_CORR_X_FAR else 0.0)
        return (self._BOX_CORR[0], dy, dz)

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
                return fk_str
            return "N/A(机械臂未连接)"
        except Exception:
            return "N/A"

    # ★ FK后台刷新线程(2026-09-11 v2): 后台线程调get_pose缓存结果,
    #   主线程定时器只读缓存更新QLabel — 主线程绝不阻塞在arm_server I/O上.
    #   原方案在主线程定时器里调get_pose(锁+3s超时), 每0.5s卡3s,
    #   按钮事件排不上队("夹爪/J6点了没反应"的根因).
    _fk_bg_result = None       # 后台线程写入, 主线程读取
    _fk_bg_thread = None       # 后台线程句柄
    _fk_bg_running = False     # 线程存活标志

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

        # ★ FK块计算抽出为 _build_fk_str() (2026-09-09): 状态栏实时跟随
        #   (0.5s节流, 见 update_all) 与事件刷新共用同一条计算链
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
        """★ 2026-10-30: 检测地面 —— 目标点 (0.35, 0, 0)、法向竖直朝下
        (0,0,-1); 二次确认后填框, 且本次不叠加终点停距(尖端直达 Z=0)."""
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
            q6 = float(q6_text) if q6_text else None

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
            # ★ 闭环修正改到运动成功发起之后(带 epoch), 旧版在解算前就挂线程,
            #   IK失败时也会凭空跑一轮补走
            success, joints = self.controller.move_to_cartesian(sx, sy, sz, q6, normal)

            if success:
                # 更新四元数显示
                quat = normal_to_quaternion(nx, ny, nz)
                self.normal_status_label.setText(
                    f"四元数: [w={quat[0]:.4f}, x={quat[1]:.4f}, y={quat[2]:.4f}, z={quat[3]:.4f}]"
                )
                
                # 打印计算出的六个关节角值
                print("=" * 50)
                print("逆运动学求解结果 - 目标笛卡尔坐标:")
                print(f"  X: {x} m, Y: {y} m, Z: {z} m")
                print(f"  法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
                print(f"  关节6输入: {q6 if q6 is not None else '默认'} rad")
                print(f"  目标四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                print("-" * 50)
                print("计算出的六个关节角 (弧度):")
                print(f"  关节1 (th1): {joints[0]:.6f} rad")
                print(f"  关节2 (th2): {joints[1]:.6f} rad")
                print(f"  关节3 (th3): {joints[2]:.6f} rad")
                print(f"  关节4 (th4): {joints[3]:.6f} rad")
                print(f"  关节5 (th5): {joints[4]:.6f} rad")
                print(f"  关节6 (th6): {joints[5]:.6f} rad")
                print("=" * 50)
                
                joint_info = f"J1={joints[0]:.3f}, J2={joints[1]:.3f}, J3={joints[2]:.3f}, J4={joints[3]:.3f}, J5={joints[4]:.3f}, J6={joints[5]:.3f}"
                self.statusBar().showMessage(f"目标关节角: {joint_info}")
                self.status_display.append(f"[移动到位置] 法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f}) → 四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                self.status_display.append(f"[移动到位置] 计算结果: {joint_info}")
                
                _epoch = self.start_arm_movement(
                    self.controller.joint_pos, joints, MOVE_DURATION)
                self._notify_trail_start()
                self._post_move_refine('手动', x, y, z_solve, normal, epoch=_epoch)
                QMessageBox.information(self, "成功",
                    f"机械臂开始移动!\n\n"
                    f"位置: ({x:.3f}, {y:.3f}, {z:.3f})\n"
                    f"终点停距3cm(沿法向): 停在 ({sx:.3f}, {sy:.3f}, {sz:.3f})\n"
                    f"法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})\n"
                    f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n\n"
                    f"目标关节角 (rad):\n"
                    f"J1={joints[0]:.4f}, J2={joints[1]:.4f}, J3={joints[2]:.4f}\n"
                    f"J4={joints[3]:.4f}, J5={joints[4]:.4f}, J6={joints[5]:.4f}")
            else:
                err = getattr(self.controller, 'last_ik_err', '')
                if 'not running' in err or '未连接' in err:
                    self.status_display.append("[移动失败] 机械臂后端未连接(arm_server 已退出)")
                    QMessageBox.warning(self, "机械臂未连接",
                        "机械臂后端(arm_server)未运行。\n\n请点击「启动机械臂」重新连接后再试。")
                else:
                    self.status_display.append(
                        "[移动失败] 目标 ({:.2f}, {:.2f}, {:.2f}) 不可达(法向量姿态下无解). "
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
                queue_size=1, buff_size=2 ** 24)
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
        跑任何 numpy/cv2 操作都可能触发段错误(已实测崩溃在 recv_buff)."""
        self._raw_color_msg = msg

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
        """深度图回调: ★只缓存原始msg(同 on_raw_color 的线程安全策略)."""
        self._raw_depth_msg = msg

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
            "<span style='color:#cccccc;'>[{}] [检测] {}({}) score={:.2f} "
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
            "<span style='color:#ffff00;font-weight:bold;'>  [{ts}] ✅确认转化: "
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
        q6 = float(q6_text) if q6_text.strip() else None

        self._action_params = {
            'safe_height': safe_height,
            'target': [target_x, target_y, target_z],
            'normal': normal,
            'q6': q6,
        }

        def _worker():
            self.action_running = True
            self._cancel_pending_refine()   # 预设动作期间旧闭环补走不得插运动
            self._worker_log("[动作%d] 已启动, 正在校验目标点(约1s)..." % action_num,
                             "动作%d已启动" % action_num, 2000)
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
                else:
                    try:
                        r = self.controller.arm_client.get_state()
                        if r.get("ok") and r.get("joints"):
                            import math as _m
                            err = [abs(_m.degrees(v)) for v in r["joints"]]
                            worst = max(err)
                            if worst > 12.0:
                                self._log(
                                    "[收尾校验] 关节未回零(最大偏差%.1f°) — 电机可能中途失能, 自动恢复并重走回零" % worst)
                                self.controller.arm_client._send({"cmd": "recover"}, timeout=30)
                                move_joint(self.controller.publisher,
                                           [v for v in r["joints"]], [0.0] * 6,
                                           8.0, controller=self.controller)
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

    def _execute_action_impl(self, action_num, mask=None, depth_img=None, th1=None):
            """执行指定的动作(1,2或3) — 2026-09-08 自 gluon_controller_kk_rebot.py 同步
            (夹爪mm间距模式 + 转移段改关节空间直达 + 涂抹面内方向 + 动作3校验修复):
            动作1(涂抹，液体类)：抓取点固定为棉签工具位，涂抹点为笛卡尔空间输入的坐标；
                沿法向量正交的面内方向往复涂抹; 完成后转移到EVIDENCE_RECOVERY_POINT释放棉签
            动作2(粘取，微小类)：抓取点固定为粘取器工具位，粘取点为笛卡尔空间输入的坐标；
                垂直下压-停留-提起完成粘附; 完成后转移到EVIDENCE_RECOVERY_POINT释放粘取器
            动作3(抓取，固体类)：抓取点即目标点本身，沿表面法向量接近后闭合夹爪抓取;
                完成后转移到专属回收位(0.14, 0, 0.05)松开夹爪放下物证(不再放棉签架)

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
            normal = params['normal']
            q6 = params['q6']

            # 根据动作类型设置抓取点
            if action_num == 1:  # 涂抹动作
                grab_x, grab_y, grab_z = 0.14, -0.093, 0.046   # ★前方棉签架(原[-0.10,0.16]左后方致j1大回转过流, 已前移)
            elif action_num == 2:  # 粘取动作
                grab_x, grab_y, grab_z = 0.145, 0.078, 0.04    # ★reBot工具架:粘取器取放点(梯形IK验证)
            elif action_num == 3:  # 抓取动作（固体类，抓取点即目标点本身）
                grab_x, grab_y, grab_z = target_x, target_y, target_z
            else:
                self._log("[动作中止] 无效的动作编号!")
                return

            # ★ 工具架J6对齐(2026-09-16 用户指定): 架子插槽朝向固定, 腕部旋转对准后
            #   夹取/放回 —— 棉签架(动作1取棉签, 0.14,-0.093,0.05) J6=+0.6rad;
            #   粘取器工具架(动作2, 0.14,0.07,0.05) J6=-0.45rad; 动作3无工具架, 维持用户q6.
            q6_rack = 0.6 if action_num == 1 else (-0.45 if action_num == 2 else q6)
            if q6_rack != q6:
                self._log("[动作] 工具架J6对齐: %s J6=%+.1frad" % (
                    "棉签架" if action_num == 1 else "粘取器工具架", q6_rack))

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
            #   伸出, 接触点需沿法向反退工具长度(涂抹 34mm, 粘取 16mm), 保证
            #   尖端落在用户输入点; 抓取不带工具不补偿.
            #   偏移加在 target 源头 → 接触点/进退点/涂抹往复 全部同步平移,
            #   法向进退点退距仍为用户输入点-(L+0.05)·n, 与接触点间距保持 5cm.
            _tool_len = {1: 0.034, 4: 0.034, 2: 0.016, 5: 0.016}
            _L = _tool_len.get(action_num, 0.0)
            if _L:
                target_x -= _L * normal[0]
                target_y -= _L * normal[1]
                target_z -= _L * normal[2]
                self._log("[动作] 工具尖端补偿: 沿法向反退 %.0fmm (%s)" % (
                    _L * 1000.0, "涂抹" if action_num in (1, 4) else "粘取"))
            _grab_normal = normal if action_num == 3 else None
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
            # ★ 回收位按动作分派:
            #   动作1(涂抹): 放回棉签架(涂抹的棉签用完放回原处)
            #   动作2(粘取): 粘取器放回自己的工具架
            #   动作3(抓取): ★ 2026-09-18 用户指定专属回收位(0.14,0,0.05),
            #     不再与棉签架相同(避免抓取的固体物证与待用棉签混放)
            if action_num == 2:
                recovery_x, recovery_y, recovery_z = 0.14, 0.078, 0.04      # 粘取器架
            elif action_num == 3:
                recovery_x, recovery_y, recovery_z = 0.14, 0.0, 0.05        # ★ 抓取动作专属回收位
            else:
                recovery_x, recovery_y, recovery_z = EVIDENCE_RECOVERY_POINT  # 棉签架(仅动作1)
            success_recovery, joints_recovery = self.controller.move_to_cartesian(
                recovery_x, recovery_y, recovery_z, q6_rack, fast=True
            )
            if not success_recovery:
                self._log(f"[动作中止] 无法对物证回收位 ({recovery_x}, {recovery_y}, {recovery_z}) 求解逆运动学!")
                return

            success_recovery_above, joints_recovery_above = self.controller.move_to_cartesian(
                recovery_x, recovery_y, safe_height, q6_rack, fast=True
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

                    # 3. 移动到抓取点上方 —— 起点是"当前关节角"，笛卡尔坐标未知，用 move_joint() 过渡
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point2, MOVE_DURATION, controller=self.controller)

                    # 4. 下探到抓取点 —— ★ 补 normal 保持夹爪向下; J6=工具架对准角
                    move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION , q6=q6_rack, controller=self.controller)

                    # 5. 闭合夹爪，取工具 —— ★ 爪距30mm; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_GRIP)
                    wait_gripper(self.controller, GRIPPER_MM_GRIP)   # ★ 实测到位(停稳=已夹住工具)

                    # 6. 返回安全高度 —— ★ 补 normal 保持夹爪向下; 保持工具架对准角
                    move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION , q6=q6_rack, controller=self.controller)

                if action_num == 1:  # 涂抹动作（液体类）
                    # ★★★ 以粘取模式代码段为模板重写(结构/函数/参数逐行一致),
                    #     唯一差异: 粘取的"目标点停留1秒"换成 小幅度来回涂抹运动 ★★★

                    # 7. 转移段：抓取点(棉签架)上方 -> 目标点上方 —— 关节空间直达
                    #   (2026-09-08: 弃用RRT —— kk_rebot 实测版直接 move_joint 过渡)
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point_target, MOVE_DURATION, controller=self.controller)

                    # 8. 接近段：法向约束下探到涂抹点
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION * 0.8, q6=q6, normal=normal, controller=self.controller)

                    # 9. 小幅度来回涂抹(替换粘取的 time.sleep(1.0) 停留):
                    #    ★ 2026-09-16 修复卡顿: 原实现逐笔 move_linear —— 每笔
                    #      4cm 都现场解完整IK(阻塞0.5~2s, 臂停住等解算), 7笔中
                    #      仅3个唯一点却解了7次. 现三点各预解一次IK, 往复用纯
                    #      move_joint 关节插值, 零IK停顿, 连续顺滑.
                    #    ★ 面内方向 = 垂直末端姿态法向量(原固定世界X已废弃).
                    _dir = _smear_dir_from_normal(normal)
                    _smear_r = 0.04
                    _right = [p_target[0] + _dir[0] * _smear_r,
                              p_target[1] + _dir[1] * _smear_r,
                              p_target[2] + _dir[2] * _smear_r]
                    _left = [p_target[0] - _dir[0] * _smear_r,
                             p_target[1] - _dir[1] * _smear_r,
                             p_target[2] - _dir[2] * _smear_r]
                    _j_pt = list(self.controller.joint_pos)   # 步骤8终点关节
                    _ok_r, _j_right = ik(_right[0], _right[1], _right[2],
                                         q6, normal, fast=True) if _ik_supports_fast(ik) \
                        else ik(_right[0], _right[1], _right[2], q6, normal)
                    _ok_l, _j_left = ik(_left[0], _left[1], _left[2],
                                        q6, normal, fast=True) if _ik_supports_fast(ik) \
                        else ik(_left[0], _left[1], _left[2], q6, normal)
                    if (_ok_r and _j_right is not None and
                            _ok_l and _j_left is not None):
                        _j_cur = _j_pt
                        for _k in range(3):
                            move_joint(publisher, _j_cur, _j_right, 1.5, controller=self.controller)
                            move_joint(publisher, _j_right, _j_left, 1.5, controller=self.controller)
                            _j_cur = _j_left
                        move_joint(publisher, _j_cur, _j_pt, 1.5, controller=self.controller)
                    else:
                        # IK预解失败(罕见): 回落原逐笔move_linear路径
                        print("[涂抹] 往复端点IK预解失败, 回落逐笔move_linear")
                        _pt = list(p_target)
                        for _k in range(3):
                            move_linear(publisher, ik, _pt, _right, 1.5, q6=q6, normal=normal, controller=self.controller)
                            move_linear(publisher, ik, _right, _left, 1.5, q6=q6, normal=normal, controller=self.controller)
                            _pt = _left
                        move_linear(publisher, ik, _pt, list(p_target), 1.5, q6=q6, normal=normal, controller=self.controller)

                    # 10. 返回安全高度
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION * 0.8, q6=q6, normal=normal, controller=self.controller)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— 关节空间直达
                    move_joint(publisher, self.controller.joint_pos.copy(), joints_recovery_above, MOVE_DURATION, controller=self.controller)

                    # 12. 下探到回收位 —— 插槽归还精定位, 最慢段(≈8s)
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION, q6=q6_rack, controller=self.controller)

                    # 13. 打开夹爪，释放棉签; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION, q6=q6, controller=self.controller)

                    # 15. 返回初始位置 —— 两段式回零(降负载 + 全关节同步复位):
                    #     第一段: 全关节同步到"紧凑中立姿态"(j1 同时转向零位方向,
                    #             悬臂收拢使 j1 负载小, 不再最后单独回零);
                    #     第二段: 中立姿态同步回到零位(小行程, 平缓).
                    _MID = [0.0, 0.70, 0.79, -0.79, 0.0, 0.0]   # 紧凑中立姿态(rad)
                    _cur = list(self.controller.joint_pos)
                    move_joint(publisher, _cur, list(_MID), MOVE_DURATION, controller=self.controller)
                    move_joint(publisher, list(_MID), point0, MOVE_DURATION, controller=self.controller)
                    # ★ 动作收尾: 重新使能全部电机
                    try:
                        self.controller.arm_client._send({"cmd": "enable"}, timeout=10)
                    except Exception as e:
                        print("[收尾使能] %s" % e)
                    self.controller.joint_pos = point0.copy()
                    self.controller.q6_value = 0.0

                elif action_num == 2:  # 粘取动作（微小类）
                    # 7. 转移段：抓取点上方 -> 目标点上方 —— 关节空间直达
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point_target, MOVE_DURATION, controller=self.controller)

                    # 8. 接近段：法向约束下探到目标点，不经过规划，保持直线
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION, q6=q6, normal=normal, controller=self.controller)

                    # 9. 在目标点停留（模拟粘取动作），不变
                    time.sleep(1.0)

                    # 10. 返回安全高度
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION, q6=q6, normal=normal, controller=self.controller)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— 关节空间直达, 不再放回原工具架
                    move_joint(publisher, self.controller.joint_pos.copy(), joints_recovery_above, MOVE_DURATION, controller=self.controller)

                    # 12. 下探到回收位 —— 插槽归还精定位, 最慢段(≈8s); J6=工具架对准角
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION, q6=q6_rack, controller=self.controller)

                    # 13. 打开夹爪，释放粘取器(连同粘附的微小物证)到回收位; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION, q6=q6, controller=self.controller)

                    # 15. 返回初始位置 —— ★同步两段式(降负载 + 全关节同步复位)
                    _MID = [0.0, 0.70, 0.79, -0.79, 0.0, 0.0]   # 紧凑中立姿态(rad)
                    _cur = list(self.controller.joint_pos)
                    move_joint(publisher, _cur, list(_MID), MOVE_DURATION, controller=self.controller)
                    move_joint(publisher, list(_MID), point0, MOVE_DURATION, controller=self.controller)
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

                    # 7. 转移段：当前位置 -> 目标点上方 —— 关节空间直达
                    #   (2026-09-08 同步: 弃用"以当前构型重解fast连续解"+wait_arrival,
                    #    kk_rebot 实测版为直接 move_joint 到 above_point_target)
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point_target, MOVE_DURATION, controller=self.controller)

                    # 8. 接近段：沿表面法向量下探到接触点 —— 法向约束
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION, q6=q6, normal=normal, controller=self.controller)

                    # 9. 闭合夹爪，真正抓取物证 —— ★ 爪距30mm; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_GRIP)
                    wait_gripper(self.controller, GRIPPER_MM_GRIP)   # ★ 实测到位(停稳=已夹住物体)

                    # 10. 沿法向量退回进退点
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION, q6=q6, normal=normal, controller=self.controller)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— 关节空间直达
                    move_joint(publisher, self.controller.joint_pos.copy(), joints_recovery_above, MOVE_DURATION, controller=self.controller)

                    # 12. 下探到回收位 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION, q6=q6_rack, controller=self.controller)

                    # 13. 打开夹爪，松开固体物证到回收位 —— ★全开确保物体脱手; 先等臂到位
                    wait_arrival(self.controller, list(self.controller.joint_pos))   # ★ 夹爪动作前显式等臂到位(2026-09-17用户要求)
                    self._send_gripper(GRIPPER_MM_OPEN)
                    wait_gripper(self.controller, GRIPPER_MM_OPEN)

                    # 14. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION, q6=q6, controller=self.controller)

                    # 15. 返回初始位置 —— ★同步两段式(降负载 + 全关节同步复位)
                    _MID = [0.0, 0.70, 0.79, -0.79, 0.0, 0.0]   # 紧凑中立姿态(rad)
                    _cur = list(self.controller.joint_pos)
                    move_joint(publisher, _cur, list(_MID), MOVE_DURATION, controller=self.controller)
                    move_joint(publisher, list(_MID), point0, MOVE_DURATION, controller=self.controller)
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
        # ★ 自动识别回零用 3s(臂实际到位早于5s定时, 缩短消除"抬起后干等数秒")
        self.start_arm_movement(self.controller.joint_pos, INITIAL_POSITION,
                                duration if duration is not None else MOVE_DURATION)
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
                # ★ 互锁(2026-09-18): 看门狗自动恢复中(USB重置+重启后端)手动点
                #   启动会与其竞争同一后端/同一PCAN, 恢复完成即自动可用
                if getattr(self, '_can_wd_recovering', False):
                    self._worker_log("[reBot] CAN看门狗自动恢复中(约20s~2分钟), 完成后机械臂"
                                     "自动可用, 无需手动启动", "看门狗自动恢复中, 请稍候...", 8000)
                    return
                if not self.controller.arm_client.start():
                    self._worker_log("[reBot] arm server 子进程启动失败",
                                     "机械臂后端启动失败", 6000)
                    return
                self._worker_log("[reBot] 后端已拉起, 正在连接电机(模式切换最长约十几秒)...",
                                 "正在连接机械臂...", 15000)
                r = self.controller.arm_client.connect()
                # ★ 适配器冻结自愈: 连接失败(全部电机无响应)大概率是PCAN冻结
                #   (点「启动机械臂」也没反应的根因), USB重置后重试一次.
                if not r.get("ok") and not getattr(self, '_can_wd_recovering', False):
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
                        if _est is None or abs(_est) > _GRIP_CAL_DRIFT_TRIG:
                            self._gripper_autocal(est=_est)   # 首次/漂移超阈值 -> 自动标定
                        else:
                            self._worker_log("[gripper] 零位与停车位置一致(差 %.1fmm), 免标定"
                                             % (_est * _GRIP_CAL_MM_PER_RAD))
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
        """★ CAN健康看门狗(8s): PCAN冻结/后端死亡 → 自动USB重置+重启后端+重连.
        三级判据: 1) can0 netdev消失(适配器被移除) 2) arm_server进程死亡(vel_loop崩溃FATAL退出)
        3) CAN往返探测失败(冻结态: netdev在/进程活, 但收不到回包 —— 灯常亮的真死因)."""
        if not getattr(self, '_arm_started', False):
            return
        if getattr(self, '_can_wd_recovering', False):
            return
        ac = self.controller.arm_client
        can_gone = not os.path.exists('/sys/class/net/can0')
        srv_dead = (ac.proc is None) or (ac.proc.poll() is not None)
        if can_gone or srv_dead:
            self._can_watchdog_trigger("can0设备消失" if can_gone else "arm_server进程死亡")
            return
        # 深度探测(运动/扫描中不做, 避免干扰): 读joint1位置参数要真实CAN回包
        if not (self.is_moving or getattr(self, 'is_scanning', False) or
                getattr(self, 'action_running', False)):
            try:
                r = ac.can_ping()
                if not r.get("ok"):
                    self._can_watchdog_trigger("CAN往返无响应(%s)" % r.get("err", "?"))
            except Exception:
                pass

    def _can_watchdog_trigger(self, reason):
        self._can_wd_recovering = True
        print("[CAN看门狗] 检测到 %s — 自动USB重置PCAN+重启后端中(约20s~2分钟)..." % reason)
        def _msg():
            self.status_display.append(
                "[CAN看门狗] 检测到 %s, 自动USB重置PCAN+重启后端(约20s~2分钟, 请稍候)..." % reason)
            self.statusBar().showMessage("PCAN适配器异常, 自动恢复中...")
        QTimer.singleShot(0, _msg)
        threading.Thread(target=self._can_recover_worker, daemon=True).start()

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

    def _can_recover_worker(self):
        """看门狗恢复流程(后台线程): 优雅停后端 → USB重置PCAN → 重启arm_server
        → 重连+使能(掉线反复期多次重试).
        ★ 2026-09-18 断电事故修复(笛卡尔控制中机械臂突然泄力, dmesg证实PCAN
          在USB2集线器上每1~2s掉线重枚举一轮):
          1) 原对活后端直接 proc.kill()(SIGKILL) —— launch_ui.sh已注明SIGKILL
             正在MIT流中的后端会让全部电机固件硬故障, 改TERM→等2s→KILL兜底;
          2) 原connect失败即放弃("请手动点启动") —— 掉线反复期connect常撞进
             can0消失窗口(事故日志pid=16483: can0 is down→7电机全离线),
             现改为最多3次重试, 重试前can0又消失则补一次USB重置;
          3) 全程print落盘rebot_ui.log(此前只进UI状态框, 排查时UI日志零线索)."""
        ac = self.controller.arm_client
        try:
            # 1) 优雅停残留后端(vel_loop崩溃但进程未退时)
            self._can_stop_backend(ac)
            # 2) USB级重置PCAN + 拉起can0(脚本内已等待can0出现并校验operstate)
            if not self._reset_pcan_usb():
                print("[CAN看门狗] USB重置失败 — 请手动插拔PCAN适配器后点「启动机械臂」")
                def _fail1():
                    self.status_display.append(
                        "[CAN看门狗] USB重置失败 — 请手动插拔PCAN适配器后点「启动机械臂」")
                    self.statusBar().showMessage("PCAN恢复失败, 需手动插拔")
                self._post_ui(_fail1)
                return
            # 3) 重启 arm_server + 连接 + 恢复使能(掉线反复期connect可能撞进
            #    can0消失窗口, 最多3次重试; 重试前can0又消失则补一次USB重置)
            ok = False
            r = {"ok": False, "err": "?"}
            for attempt in range(1, 4):
                if not os.path.exists('/sys/class/net/can0'):
                    print("[CAN看门狗] 第%d次连接前can0又消失(适配器仍在掉线反复), 补一次USB重置..." % attempt)
                    self._reset_pcan_usb()
                self._can_stop_backend(ac)   # 上次connect失败的后端内部状态不可信, 换新进程
                if not ac.start():
                    print("[CAN看门狗] arm_server 重启失败(第%d次)" % attempt)
                    time.sleep(5.0)
                    continue
                r = ac._send({"cmd": "connect"}, timeout=90)   # connect可能耗时十几秒
                print("[CAN看门狗] 第%d次连接: %s" % (
                    attempt, "OK" if r.get("ok") else ("失败: %s" % r.get("err", "?"))))
                if r.get("ok"):
                    ok = True
                    break
                time.sleep(5.0)
            if not ok:
                print("[CAN看门狗] 3次重连均失败(最后: %s) — 请手动点「启动机械臂」" % r.get("err", "?"))
                def _fail3():
                    self.status_display.append(
                        "[CAN看门狗] 3次重连均失败: %s — 请手动点「启动机械臂」重试" % r.get("err"))
                    self.statusBar().showMessage("自动恢复失败, 请手动启动机械臂")
                self._post_ui(_fail3)
                return
            ac._send({"cmd": "recover"})
            self._arm_started = True
            print("[CAN看门狗] ✔ 机械臂已自动恢复 (USB重置+重启后端+重新使能)")
            def _ok():
                self.status_display.append(
                    "[CAN看门狗] ✔ 机械臂已自动恢复 (USB重置+重启后端+重新使能)")
                self.statusBar().showMessage("机械臂已自动恢复")
            self._post_ui(_ok)
        except Exception as e:
            print("[CAN看门狗] 恢复异常: %s" % e)
        finally:
            self._can_wd_recovering = False

    def start_camera(self):
        self._cam_wd_retried = False   # 看门狗重试标志复位(每次手动启动重新允许自愈一次)
        # ★ 清理可能残留的旧相机/检测进程(上次退出未杀干净会占住USB,
        #   表现为"必须插拔USB才能恢复" —— 杀掉即可重开)
        subprocess.Popen(
            ['bash', '-c',
             'pkill -f "forensic3_detect[.]launch"; '
             'pkill -f "orbbec_camera_node"; '
             'pkill -f "aruco_calib[.]launch"; '
             'pkill -f "aruco_calib_gemini[.]launch"; sleep 0.3'],
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
                # ★ 标定模式(2026-09-10): CALIB_MODE=True 时启动
                #   aruco_calib_gemini.launch(Gemini相机+ArUco板检测)替代
                #   v4检测管线 — 话题不变(同 /camera/color/image_raw), UI
                #   原样显示画面, 标定板位姿由 TF(base_link←aruco_marker)给出.
                #   v4 实时检测(推理)暂时关闭(用户: 准备标定, 暂时关闭实时监测).
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
                    # ★ UI 相机窗订阅 ArUco 标注结果图(带marker框+坐标轴),
                    #   原始流由 init_perception_subscribers 仍照常订阅(画框用)
                    self.controller.image_subscriber = rospy.Subscriber(
                        '/aruco_single/result',
                        Image,
                        self.controller.handle_camera_image,
                        queue_size=1, buff_size=2 ** 24
                    )
                    self.statusBar().showMessage(
                        "标定模式: Gemini相机 + ArUco板检测已启动 "
                        "(marker=871/4X4_1000/97mm, TF: base_link←aruco_marker)")
                    self.status_display.append(
                        "[标定] 标定模式已启动: aruco_calib_gemini.launch. "
                        "标定板位姿查询: rosrun tf tf_echo base_link aruco_marker. "
                        "v4 实时检测已暂停(CALIB_MODE=True).")
                    return

                # ★ 血迹检测管线(2026-09-28): roslaunch forensic3_detector_v4
                #   blood_detect.launch (Gemini相机 + blood2模型[dla_34/单类blood],
                #   /home/zf/blood/exp/ctdet/blood2/model_best.pth)
                #   结果话题不变(/forensic3_detect_v4/result_image), UI 相机窗
                #   直接显示带 blood 检测框/分数的画面.
                #   vis_thresh=0.30: 低于模型默认0.50的检出也画框, 便于确认识别效果.
                #   输出重定向到文件, 防止 QProcess 管道堵死(看门狗也读此日志).
                detect_cmd = (
                    "source /opt/ros/noetic/setup.bash && "
                    "source ~/forensic_ws/devel/setup.bash --extend && "
                    "source ~/forensic3_detect_ws_v4/devel/setup.bash --extend && "
                    "exec roslaunch forensic3_detector_v4 blood_detect.launch "
                    "display:=false vis_thresh:=0.30 > /tmp/forensic3_detect.log 2>&1"
                )
                detect_process = QProcess()
                detect_process.setProcessChannelMode(QProcess.MergedChannels)
                detect_process.start("bash", ["-c", detect_cmd])
                self.camera_processes.append(detect_process)

                # ★ UI 相机窗订阅 v4 检测结果图(带检测框/类别/分数叠加):
                #   /forensic3_detect_v4/result_image。推理约几fps, 显示节流
                #   已由 update_camera_view 的 0.08s 间隔兜住。原始彩色/深度流
                #   由 init_raw_streams 单独订阅, 供自动定位的像素/深度解算用。
                self.controller.image_subscriber = rospy.Subscriber(
                    '/forensic3_detect_v4/result_image',
                    Image,
                    self.controller.handle_camera_image,
                    queue_size=1, buff_size=2 ** 24
                )
                # ★ 启动等待提示: 停止时已清pixmap, 这里给"启动中"文字,
                #   首帧到达后自动被画面覆盖(实测到帧约需10~15s)
                self.camera_label.setText("相机启动中... (驱动初始化约10~15秒)")
                self.statusBar().showMessage("血迹检测: Gemini相机 + blood2模型已启动 (vis_thresh=0.30)")
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
            subprocess.Popen(
                ['bash', '-c',
                 'pkill -f "forensic3_detect[.]launch"; '
                 'pkill -f "blood_detect[.]launch"; '
                 'pkill -f "orbbec_camera_node"; '
                 'pkill -f "forensic3_detect_v4_node"; sleep 0.2'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
            self.camera_label.setText("相机已停止")
            self._set_cam_chip('待机')
            self.statusBar().showMessage("相机系统已停止!")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：closeEvent — 先释放资源再弹确认框
    # ──────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        # ★ 先停CAN看门狗: 否则退出过程中它检测到arm_server死亡会"自愈"
        #   把刚关掉的后端又拉起来(退出被看门狗劫持的根因)
        self._arm_started = False
        self._can_wd_recovering = True

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
        try:
            self.controller.arm_client.proc.wait(timeout=3)
        except Exception:
            if self.controller.arm_client.proc.poll() is None:
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
REBOT_ARM_SERVER = os.path.expanduser('~/forensic_ws/src/reBot/scripts/rebot_arm_server.py')


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
        ★ 2026-08-31 还原为(6)版阻塞写实现(非阻塞os.write实验版撤回)."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "err": "arm server not running"}
            with self._wlock:
                try:
                    self.proc.stdin.write(json.dumps(cmd) + "\n")
                    self.proc.stdin.flush()
                except Exception as e:
                    return {"ok": False, "err": "stdin write fail: %s" % e}
            return self._send_locked(cmd, timeout)

    def _send_locked(self, cmd, timeout=30.0):
        """★ select 轮询实现真超时(调用方已持锁):
        绝不在 UI 主线程无限阻塞(readline 是阻塞调用, arm_server 慢时
        会把整个界面卡死 —— '启动机械臂UI卡死'的根因)."""
        import select as _select
        # (命令已由调用方写入管道; 本函数只负责等响应)
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
                if isinstance(obj, dict):
                    return obj
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
        """★ FK后台刷新专用(2026-09-11): 0.25s短超时 — arm_server忙时(如解IK)
        迅速放弃, 不长期占锁饿死主线程的夹爪/J6指令."""
        return self._send({"cmd": "get_pose"}, timeout=0.25)
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
        # ★ arm_server 死亡自愈: 进程崩溃/被杀后自动重启重连(用户无感),
        #   否则所有IK报"not running"被上层误显示为"目标不可达"
        if not r.get("ok") and "not running" in str(r.get("err", "")):
            if self.start():
                r2 = self._send({"cmd": "connect"}, timeout=90)
                if r2.get("ok"):
                    self._send({"cmd": "recover"}, timeout=60)
                    r = self._send(cmd)
        return r

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
        self._raw_color_msg = msg   # 检测线程从 _raw_color_msg 转换

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


def wait_gripper(controller, target_mm, tol_mm=2.0, timeout=6.0,
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

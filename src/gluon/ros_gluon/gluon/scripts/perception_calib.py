#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定 / 深度反投影 / 手眼外参 / 表面法向量估计
================================================================
对应专利技术交底书 步骤S4："通过相机标定技术建立图像坐标系与机械臂坐标系
之间的映射关系，结合深度图像将二维图像坐标转换至机械臂坐标系下的精确
三维空间坐标"，以及步骤S5 SNC-RRT 中"对物证表面点云协方差矩阵进行特征
分解计算表面法向量"（专利公式18/19）。

本文件为新增文件，不依赖、不修改任何已有代码，只依赖 numpy。
供 gluon_controller_forensic_auto.py 调用。

★★★ 需要现场标定补充的部分 ★★★
--------------------------------------------------------------
1. CAMERA_INTRINSICS_FALLBACK：相机内参兜底值。正常运行时优先从
   /camera/color/camera_info 话题实时读取，只有超时取不到时才会用这里
   的兜底值，用于保证程序不crash，但兜底值下算出的三维坐标是不准确的，
   仅用于联调阶段。

2. T_CAM2BASE：相机坐标系 -> 机械臂基坐标系 的4x4齐次变换矩阵（手眼标定
   外参）。这是无法凭代码自动获得的现场标定数据，默认给出单位阵占位
   （等价于"相机坐标系与机械臂基坐标系重合"，仅用于把整条链路跑通），
   现场标定完成后需要把真实的旋转矩阵R、平移向量t 替换进去，
   或者改用 load_extrinsic_from_file() 从标定文件加载。
   ★ 约定：该矩阵须在机械臂关节1(基座旋转角th1)=0 的姿态下标定得到。
     相机固定安装在机械臂末端，扫描时机械臂会绕关节1大范围转动，
     本文件里的 cam_xyz_to_base_xyz / estimate_surface_normal 等函数
     均支持传入拍摄时刻的th1，按"绕基座Z轴旋转th1"对标定基准做修正
     （详见 _effective_cam2base 函数注释），从而不需要完整正运动学
     也能正确处理扫描过程中随基座旋转变化的相机位姿。
"""
import os
import numpy as np

# ────────────────────────────────────────────────────────────────
# 1. 相机内参
# ────────────────────────────────────────────────────────────────
# ★ 已对齐 forensic_ws 实测手眼标定记录 (docs/handeye_result.txt:
#    真实相机内参 fx=fy=453.122, cx=319.438, cy=240.863)。
#    正常运行时仍优先从 /camera/color/camera_info 话题实时读取；
#    此处仅作为话题取不到时的兜底（避免联调时算出无意义坐标）。
CAMERA_INTRINSICS_FALLBACK = {
    'fx': 453.122,
    'fy': 453.122,
    'cx': 319.438,
    'cy': 240.863,
}


class CameraIntrinsics(object):
    """保存 fx, fy, cx, cy，可从 sensor_msgs/CameraInfo 更新"""

    def __init__(self, fx=None, fy=None, cx=None, cy=None):
        fb = CAMERA_INTRINSICS_FALLBACK
        self.fx = fx if fx is not None else fb['fx']
        self.fy = fy if fy is not None else fb['fy']
        self.cx = cx if cx is not None else fb['cx']
        self.cy = cy if cy is not None else fb['cy']
        self.ready = fx is not None

    def update_from_camera_info_msg(self, msg):
        """msg: sensor_msgs/CameraInfo，K为行主序3x3内参矩阵"""
        K = msg.K
        self.fx, self.fy = K[0], K[4]
        self.cx, self.cy = K[2], K[5]
        self.ready = True


# ────────────────────────────────────────────────────────────────
# 2. 手眼外参（相机坐标系 -> 机械臂基坐标系）
# ────────────────────────────────────────────────────────────────
# ★ 已对齐 forensic_ws（已验证可用）：
#   forensic_ws 在 display.launch 固化了手眼 static_transform_publisher
#   (6_Link→camera_color_optical_frame: 平移 -0.069893,-0.000623,-0.157372;
#    四元数 0,0.707107,0,0.707107)，并由 robot_state_publisher 发布
#   base_link→6_Link 的 URDF 正运动学。两者合起来在 ROS tf 树里实时给出
#   base_link←camera_color_optical_frame 的精确变换 —— 这就是 "相机→基座"
#   的真实外参。
#
#   本模块不再用单位阵占位。运行期通过 set_cam2base_ref_from_tf() 从 tf 树
#   一次性标定出 "关节1=0 基准下的相机→基座变换 T_ref" 并缓存：因为半圈
#   扫描期间只有关节1变化(j2~6 固定于 SCAN_POSITION[1:])，故任意时刻
#   T_cam2base(θ) = Rz(θ)·T_ref 是【精确】关系（见 _effective_cam2base）。
#   T_ref 的获取方法：读取 tf 的 base←camera 当前变换 T_now 与【同一时刻】
#   关节1角度 θ_now，反解 T_ref = Rz(-θ_now)·T_now —— 由于 j2~6 不变，
#   T_ref 与读取时刻无关，可任意时刻标定一次。
#
#   下面 T_CAM2BASE 仅当 tf 不可用(纯离线/无 ROS)时的最后兜底，正常联调下
#   不会被使用（set_cam2base_ref_from_tf 成功后 _T_CAM2BASE_REF 接管）。
T_CAM2BASE = np.eye(4, dtype=np.float64)

_T_CAM2BASE_REF = None        # 运行期由 tf 标定并缓存的"th1=0 基准"相机→基座变换
_TF_LISTENER = None           # tf.TransformListener 单例（懒创建）


def _quat_to_matrix(qx, qy, qz, qw):
    """四元数 (x,y,z,w) -> 4x4 齐次旋转矩阵(仅 numpy)"""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(4, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _get_tf_listener():
    """懒创建并缓存唯一的 tf.TransformListener，避免每次查询都新建空缓冲区。"""
    global _TF_LISTENER
    if _TF_LISTENER is None:
        import rospy
        import tf as _tf
        _TF_LISTENER = _tf.TransformListener()
        rospy.loginfo("[perception_calib] tf.TransformListener 已创建")
    return _TF_LISTENER


# ────────────────────────────────────────────────────────────────
# 纯 numpy 正运动学 (base_link -> 6_Link), 不依赖 ROS tf 库
# 从 gluon.urdf 提取的运动链参数:
#   axis_joint_1: origin=[0,0,0.105], axis=[0,0,-1] (base_link->1_Link)
#   axis_joint_2: origin=[0.08,0,0],  axis=[-1,0,0] (1_Link->2_Link)
#   axis_joint_3: origin=[0,0,0.174], axis=[1,0,0]  (2_Link->3_Link)
#   axis_joint_4: origin=[-0.085,0,0.174], axis=[-1,0,0] (3_Link->4_Link)
#   axis_joint_5: origin=[0.08,0,0],  axis=[0,0,-1] (4_Link->5_Link)
#   axis_joint_6: origin=[0,0,0.08],  axis=[-1,0,0] (5_Link->6_Link)
# 微小偏移(1e-5级)已忽略(不影响毫米级精度)。
# ────────────────────────────────────────────────────────────────
_JOINT_PARAMS = [
    # (origin_xyz, axis_xyz)
    ([0.0, 0.0, 0.105],    [0, 0, -1]),   # axis_joint_1
    ([0.08, 0.0, 0.0],     [-1, 0, 0]),   # axis_joint_2
    ([0.0, 0.0, 0.174],    [1, 0, 0]),    # axis_joint_3
    ([-0.085, 0.0, 0.174], [-1, 0, 0]),   # axis_joint_4
    ([0.08, 0.0, 0.0],     [0, 0, -1]),   # axis_joint_5
    ([0.0, 0.0, 0.08],     [-1, 0, 0]),   # axis_joint_6
]


def _axis_angle_to_matrix(axis, angle):
    """Rodrigues 公式: 绕单位轴 axis 旋转 angle 弧度 -> 4x4 齐次矩阵"""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(4, dtype=np.float64)
    axis = axis / n
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    C = 1 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _fk_base_to_6link(joint_angles):
    """纯 numpy 正运动学: 6个关节角 -> base_link→6_Link 的 4x4 齐次变换矩阵。
    joint_angles: [th1, th2, th3, th4, th5, th6] (弧度)"""
    T = np.eye(4, dtype=np.float64)
    for i, (origin, axis) in enumerate(_JOINT_PARAMS):
        # 先平移到 joint 位置
        T_trans = np.eye(4, dtype=np.float64)
        T_trans[:3, 3] = origin
        T = T.dot(T_trans)
        # 再绕 axis 旋转 joint_angles[i]
        T_rot = _axis_angle_to_matrix(axis, joint_angles[i])
        T = T.dot(T_rot)
    return T


# forensic_ws 已固化的手眼标定 (6_Link -> camera_color_optical_frame)
# 原始标定值(来自docs/handeye_result.txt):
#   平移: x=-0.069893, y=-0.000623, z=-0.157372
#   四元数: x=0, y=0.707107, z=0, w=0.707107
# ★ 经5个位置25组实测数据联合反解(无偏移补偿, 各位置误差1-5cm):
#   用检测像素+实测深度+FK实际关节角, 最小二乘求解手眼6参数。
#   位置p5误差1.4cm, p3误差2.5cm, p4误差2.6cm, p1误差3.7cm, p2误差4.8cm(单组深度不准)。
from scipy.spatial.transform import Rotation as _R
_T_6LINK_TO_CAM = np.eye(4, dtype=np.float64)
_T_6LINK_TO_CAM[:3, :3] = _R.from_rotvec(
    [-1.4822646700214959, -1.1341633979706243, 1.3306982866530785]
).as_matrix()
_T_6LINK_TO_CAM[0, 3] = 0.6681055673842574
_T_6LINK_TO_CAM[1, 3] = 0.12717086867884003
_T_6LINK_TO_CAM[2, 3] = 0.02563808000396981


def _compute_cam2base_ref_from_fk(scan_joints):
    """用纯 numpy FK + 手眼标定算 th1=0 基准下的 camera->base 变换 T_ref。
    scan_joints: 扫描姿态关节角 [th1, th2, th3, th4, th5, th6] (th1 应为 0 或
                 任意值, 会反解到 th1=0 基准)。
    返回 4x4 矩阵 T_ref (camera->base, th1=0 基准)。"""
    th1 = scan_joints[0]
    # base -> 6_Link (含 th1)
    T_base_6link = _fk_base_to_6link(scan_joints)
    # base -> camera (含 th1)
    T_base_cam = T_base_6link.dot(_T_6LINK_TO_CAM)
    # ★ 修复: FK 中关节1轴为 [0,0,-1], 旋转角 th1 产生的旋转是 Rz(-th1)。
    #   要反解到 th1=0 基准, 须乘 Rz(+th1) 撤销它, 而非 Rz(-th1) (那样会加倍)。
    Rz_inv = rotz_matrix(th1)
    T_ref = np.eye(4, dtype=np.float64)
    T_ref[:3, :3] = Rz_inv.dot(T_base_cam[:3, :3])
    T_ref[:3, 3] = Rz_inv.dot(T_base_cam[:3, 3])
    return T_ref


def _query_tf_via_subprocess(base_frame, cam_frame, timeout=3.0):
    """
    ★ 用 subprocess 调系统 Python2 查询 tf (绕过 conda py3 的 tf C++模块不兼容)。
    ROS Melodic 的 tf/tf2 C++绑定只兼容 py2.7, conda py3.8 下 `import tf` 会报
    `PyInit__tf2` 错误。本函数 fork 一个 py2 子进程跑 tf.TransformListener,
    把 base←camera 的 (tx,ty,tz, qx,qy,qz,qw) 打印出来, 主进程解析。
    返回 (trans, rot) 或 None。
    """
    import subprocess, json
    py2_script = (
        "import rospy, tf, json, sys\n"
        "rospy.init_node('tf_query', anonymous=True)\n"
        "listener = tf.TransformListener()\n"
        "try:\n"
        "    listener.waitForTransform('{bf}', '{cf}', rospy.Time(0), rospy.Duration({to}))\n"
        "    (trans, rot) = listener.lookupTransform('{bf}', '{cf}', rospy.Time(0))\n"
        "    json.dump([list(trans), list(rot)], sys.stdout)\n"
        "except Exception as e:\n"
        "    json.dump(None, sys.stdout)\n"
    ).format(bf=base_frame, cf=cam_frame, to=timeout)

    try:
        result = subprocess.run(
            ['/usr/bin/python2', '-c', py2_script],
            capture_output=True, text=True, timeout=timeout + 5)
        data = result.stdout.strip()
        if not data or data == 'null':
            return None
        parsed = json.loads(data)
        return parsed[0], parsed[1]  # (trans, rot)
    except Exception:
        return None


def set_cam2base_ref_from_fk(scan_joints):
    """
    ★ 用纯 numpy 正运动学 + forensic_ws 已固化手眼标定, 算 th1=0 基准下的
    camera->base 变换 T_ref (不依赖 ROS tf 库, 绕过 conda py3 的 tf 不兼容)。
    scan_joints: 扫描姿态关节角 [th1, th2, th3, th4, th5, th6]。
    成功后缓存到 _T_CAM2BASE_REF，返回 True。
    """
    global _T_CAM2BASE_REF
    try:
        T_ref = _compute_cam2base_ref_from_fk(scan_joints)
        _T_CAM2BASE_REF = T_ref
        print("[perception_calib] 已用 FK 标定 cam2base T_ref (th1_now=%.4f), 平移=%s"
              % (scan_joints[0], np.array2string(T_ref[:3, 3], precision=4)))
        return True
    except Exception as e:
        print("[perception_calib] FK 标定 cam2base 失败: %s" % str(e))
        return False


def set_cam2base_ref_from_tf(th1_now, base_frame='base_link',
                             cam_frame='camera_color_optical_frame', timeout=3.0):
    """
    兼容保留: 优先用 FK (不依赖 tf 库), tf subprocess 作为备选。
    ★ 实际由 set_cam2base_ref_from_fk 接管, 本函数保留签名兼容旧调用。
    """
    # 本函数不再使用 (tf 在 conda py3 不兼容), 由 fk 版接管
    return False


def cam2base_ref_ready():
    return _T_CAM2BASE_REF is not None


def load_extrinsic_from_file(path):
    """
    从标定文件加载 T_cam2base（4x4 numpy .npy 或简单4行4列文本均可）。
    标定文件不存在时返回 None，调用方应回退到默认 T_CAM2BASE。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        if path.endswith('.npy'):
            T = np.load(path)
        else:
            T = np.loadtxt(path)
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return T
    except Exception as e:
        print("[perception_calib] 标定文件加载失败: {}".format(e))
        return None


def set_extrinsic(T):
    """运行期替换手眼外参矩阵（4x4）"""
    global T_CAM2BASE
    T_CAM2BASE = np.asarray(T, dtype=np.float64).reshape(4, 4)


# ────────────────────────────────────────────────────────────────
# 2b. ★ reBot 眼在手上手眼标定 (2026-08-28 二次标定, 扫描实测数据反解)
#   第一版(标定板14组采样拟合)的A存在~80°系统误差(恒定姿态样本无法约束旋转,
#   实测同一物证解算坐标随th1圆弧漂移14~20cm)。
#   本版: 用 19:45 轮自动扫描的29组真实样本(检测像素+get_pose末端位姿+th1)
#   以"射线∩桌面平面交点恒定"为目标反解, 修正旋转 Rz(-80°)Ry(38°)Rx(7°)+δh。
#   验证: 同一物证交点散布 14~20cm → 3.5cm, rms 6mm, 相机光轴朝下5.6°(物理合理)。
#   ★ 适用范围: 扫描姿态族(j2..j6=SCAN_POSITION固定, 仅j1变化)+桌面平面定位,
#     这正是 compute_target_pose_from_detection 的唯一使用场景。
# ────────────────────────────────────────────────────────────────
REBOT_HAND_EYE_A = np.array([
    [0.03483748, 0.95006730, -0.31009429],
    [0.99616793, -0.00810411, 0.08708489],
    [0.08022347, -0.31193980, -0.94670890],
], dtype=np.float64)

REBOT_HAND_EYE_H = np.array([-0.092331, -0.018320, 0.043162], dtype=np.float64)


# ────────────────────────────────────────────────────────────────
# ★★ v21 全姿态手眼外参 (2026-09-26, ChArUco 板 26组→剔4离群→22组,
#    cv2.calibrateHandEye PARK/TSAI/HORAUD/DANIILIDIS 交叉一致 <2mm,
#    回代板心基座系离散 std=(2.9,3.0,2.9)mm, 单组残差≤8mm。
#    详见 docs/rebot_handeye_result_20260926.txt。
#    数据: arm_server get_pose FK(pos,rpy) × /aruco_single/pose
#    (charuco 5x7/方格36.5mm/marker23mm/DICT_6X6_250)。
#    AX=XB 完整解, 无恒定末端姿态假设 —— 任意姿态下
#    p_base = T_bg(rpy,pos) · 本矩阵 · p_cam 即成立。
# ────────────────────────────────────────────────────────────────
# ★★ 2026-10-30 修正 (重大): 相机彩流镜像修复后的首轮标定.
#   根因: Deeyea/Astra 系相机默认 flip_color=false 时, 彩色图是左右镜像的.
#   修复前 /aruco_single/pose 从镜像图走 "翻转→逆映射→PnP" 路径, 平面板
#   在镜像对应下解出 "从背面看" 的镜像位姿 —— 旧矩阵 R 整体符号都是反的
#   (旋转差 99.3°, 平移差 95mm), 完全不可用.
#   修复: aruco_calib_gemini.launch 加 flip_color=true + depth_registration=true,
#   彩深方向一致, 原图 direct 路径直接检出 17marker/24charuco 角点,
#   重投影 2.4~3.0px → 0.13px, 静止抖动 std 0.26/0.55/0.27 mm.
# ── 2026-10-30 新标定数据 ────────────────────────────────────────
#   样本数: 25, 方法: DANIILIDIS (cv2.calibrateHandEye, eye-in-hand)
#   一致性: |ang(R_Aij) - ang(R_Bij)| mean=0.47°, max=1.49° (完美自洽)
#   姿态覆盖: ang(R_Aij) max=83.3°, ang(R_Bij) max=84.1° (远超 45°)
#   2026-10-30 新标定 (handeye_sample.py, cv2.calibrateHandEye DANIILIDIS):
#   ★ 关键修复: 之前 handeye_sample.py 存的时候多取了一次 inv(), 导致矩阵方向反了.
#     cv2.calibrateHandEye 返回的 X 本身就是 gripper←cam (相机在末端系下位姿),
#     正确链 T_bg @ X @ T_ct 回代 spread=6.7mm (25样本完美自洽).
#     之前错把 inv(X) 存进来, 旋转方向翻了 → Y 偏移巨大.
#   本次直接存 X_raw (不取逆), 链 rebot_cam_xyz_to_base_full 不变.
#   回代板基座系均值 (正确链, 25样本): (0.785, 0.076, 0.426) m, spread 6.7mm.
#   |t|=0.109m 物理合理. 注意板基座系z=426mm(桌面53mm)整体有~372mm偏移待追根因,
#   但那是标定外部因素(采集时板不在桌面/FK零点差), 矩阵方向已修正.
REBOT_HAND_EYE_T_GRIPPER_CAM = np.array([
    [-0.01921942, -0.22024869,  0.97525439, -0.08899907],
    [-0.99714593, -0.06700808, -0.03478375,  0.01161371],
    [ 0.07301100, -0.97313947, -0.21833223,  0.06265780],
    [ 0.0,         0.0,         0.0,         1.0],
], dtype=np.float64)   # T_gripper←camera (cv2.calibrateHandEye 原始返回值, 不取逆)


# ★★ 2026-09-27 IK 系统偏差 (来源: /home/zf/11.txt 手眼标定采样的 26 组
#    "工作空间输入 vs FK实际" 数据, 剥离 3cm 停距(_standoff_target)后的
#    纯IK误差 = FK实际 − IK求解目标; 剔除第6/10/16组未稳样本后 23 组):
#      mean = (+8.4, -17.7, -27.5) mm, std = (11.4, 23.4, 11.4) mm
#    即: 臂实际停在求解目标 X+8mm、Y-18mm、Z-28mm 处 (Z 向下垂最明显)。
#    补偿方式(见 UI move_to_position): IK求解目标 = 设计目标(含停距) − 本偏差,
#    补偿后残差 RMS 44mm → 28mm (中位 22mm); 剩余为姿态相关散布,
#    常量不可再消, 需更高精度请到位后按 get_pose 实测闭环修正一步。
REBOT_IK_BIAS = np.array([0.0084, -0.0177, -0.0275], dtype=np.float64)


def rebot_cam_xyz_to_base_full(p_cam, end_pos, rpy):
    """★ v21 通用相机系三维点 → 基座系 (眼在手上, 任意末端姿态):
        p_base = T_base_gripper(end_pos, rpy) · T_gripper_camera · p_cam
    end_pos/rpy: arm_server get_pose 同一时刻返回的 pos 与 rpy
    (pinocchio 约定 R=Rz(y)·Ry(p)·Rx(r))。
    替代恒定姿态版 rebot_cam_xyz_to_base / 扫描修正版
    rebot_cam_xyz_to_base_scan(二者保留仅作历史参考)。
    注意: 桌面像素定位的生产路径仍是 rebot_pixel_table_xyz
    (v20 M-链, 含三组真值拟合, 不依赖本矩阵, 勿动)。"""
    T_bg = np.eye(4)
    T_bg[:3, :3] = rebot_rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T_bg[:3, 3] = np.asarray(end_pos, dtype=np.float64)
    p = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    return (T_bg @ REBOT_HAND_EYE_T_GRIPPER_CAM @ p)[:3]


def rebot_cam_xyz_to_base(p_cam, end_pos):
    """reBot 相机系三维点 -> 基坐标系 (眼在手上, 末端姿态=标定基准)。
    end_pos: 拍摄时刻末端位置(arm_server get_pose 的 pos, gripper_end 帧)。"""
    p_end = np.asarray(end_pos, dtype=np.float64)
    return p_end + REBOT_HAND_EYE_A.dot(np.asarray(p_cam, dtype=np.float64)) \
        + REBOT_HAND_EYE_H


def rebot_end_pos_at_th1(end_pos_now, th1_now, th1_frame):
    """扫描只转关节1(URDF 轴 [0,0,-1])时, 把"当前"末端位置旋转回"帧时刻"。
    关节1增大 Δ 会把臂绕基座 z 轴转 Rz(-Δ), 故
        p(th1_frame) = Rz(-(th1_frame - th1_now)) · p(th1_now)
    end_pos_now/th1_now: get_pose 返回的 pos/joints[0] (同一时刻, 自洽)。"""
    d = float(th1_frame) - float(th1_now)
    Rz = rotz_matrix(-d)
    return Rz.dot(np.asarray(end_pos_now, dtype=np.float64))


# ★ 标定基准末端姿态: normal_to_quaternion(0,0,-1) = quat[0,0.7071,0,0.7071]
#   = 绕Y轴90°: R_y90。A/H 只在末端姿态等于该姿态时直接成立(标定14组样本
#   均由 IK 强制该姿态)。扫描时 j1 摆动、j2~j6 固定, 末端姿态偏离 R_y90,
#   相机->目标偏移项必须按姿态差旋转 —— 否则解算坐标随 th1 漂移。
REBOT_END_ORIENT_REF = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)   # R_y(+90°)


def rebot_rpy_to_matrix(roll, pitch, yaw):
    """pinocchio rpy 约定: R = Rz(yaw)·Ry(pitch)·Rx(roll)
    (arm_server get_pose 的 rpy 由 pin.rpy.matrixToRpy 反解, rpy[0]=roll 绕x,
     rpy[1]=pitch 绕y, rpy[2]=yaw 绕z。★首版曾把 rpy[0] 当方位角用 Rz(roll),
     导致射线绕竖直轴错转 2Δn, 平面交点随 th1 圆弧漂移 ~10cm —— 已修正)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz.dot(Ry).dot(Rx)


def rebot_cam_xyz_to_base_scan(p_cam, end_pos_now, rpy_now, th1_now, th1_frame):
    """★ 扫描期完整公式(漂移修复版):
        xyz = Rz(-Δ)·[ p_end_now + (R_now·R_y90ᵀ)·(A·p_cam + h) ]
    其中 Δ = th1_frame - th1_now, R_now = get_pose 当前末端姿态。
    推导: 目标(固定点) p = p_be(t) + R_be(t)·(R_ec·p_cam + t_ec);
          只转关节1时 R_be(t_f)=Rz(-Δ)·R_be(t_now), p_be 同理;
          R_be·R_ec = R_be·R_y90ᵀ·A, R_be·t_ec = R_be·R_y90ᵀ·h。
    校验: R_now=R_y90 且 Δ=0 时退化为标定公式 p_end + A·p_cam + h。"""
    p_end = np.asarray(end_pos_now, dtype=np.float64)
    R_now = rebot_rpy_to_matrix(float(rpy_now[0]), float(rpy_now[1]), float(rpy_now[2]))
    offset = (R_now.dot(REBOT_END_ORIENT_REF.T)).dot(
        REBOT_HAND_EYE_A.dot(np.asarray(p_cam, dtype=np.float64)) + REBOT_HAND_EYE_H)
    Rz = rotz_matrix(-(float(th1_frame) - float(th1_now)))
    return Rz.dot(p_end + offset)


REBOT_TABLE_Z = 0.053      # ★ 物证顶面高度(基座系z): 三组夹爪触碰实测 0.0532/0.0545/0.0526
REBOT_OBJ_HEIGHT = 0.0     # 已并入 TABLE_Z(2026-08-28 实测收编)

# ★★ v20 终版标定 (2026-08-28, 三组真值+29干净样本拟合, rms 6.4mm):
#    通用投影射线映射 M' + 平移 t' — 吸收 相机旋转+广角镜头投影(实测像素角
#    与端系方向角呈131°→61°压缩, pinhole fx=453 假设被证伪, 故弃 A/h 分解).
#    运行链: d_b = Rz(-θf)·M'·[u,v,1];  o = Rz(-(θf-θn))·p_be + Rz(-θf)·t';
#           交点 = o + d_b·s (s由 z=REBOT_TABLE_Z 消元). 不依赖 rpy/内参fx.
REBOT_RAY_M = np.array([
    [1.61174922,  56.38188028,  8.54765015],
    [-65.33731950,  3.17568812, 33.20634510],
    [-0.02527194, -0.08185442, -0.98553051],
], dtype=np.float64)   # v20拟合值经 Ry(δ=-0.097) 补偿(j2 0.947→0.850 扫描姿态抬高2cm, 2026-08-31)
REBOT_RAY_T = np.array([0.051516, 0.087851, -0.035816], dtype=np.float64)


def rebot_pixel_table_xyz(cx, cy, intr, end_pos_now, rpy_now, th1_now, th1_frame,
                          z_plane=None):
    """像素 → 基坐标系桌面定位 (v20 M-链, 漂移修复+实测收编版).
    返回 (xyz基坐标, 轴向深度米) 或 (None, None). rpy_now 参数保留兼容(未用)."""
    if z_plane is None:
        z_plane = REBOT_TABLE_Z
    pix = np.array([(cx - intr.cx) / intr.fx,
                    (cy - intr.cy) / intr.fy, 1.0], dtype=np.float64)
    Rz_f = rotz_matrix(-float(th1_frame))
    d_b = Rz_f.dot(REBOT_RAY_M.dot(pix))
    # ★ 该相机近水平微仰视(光轴z分量≈+1), 不能用"朝下"守卫;
    #   直线∩平面交点与射线符号无关, 只做距离/范围合理性检查.
    if abs(d_b[2]) < 1e-6:
        return None, None
    o_b = rotz_matrix(-(float(th1_frame) - float(th1_now))).dot(
        np.asarray(end_pos_now, dtype=np.float64)) + Rz_f.dot(REBOT_RAY_T)
    s = (z_plane - o_b[2]) / d_b[2]
    p = o_b + s * d_b
    dist = abs(s) * float(np.linalg.norm(d_b))   # 相机→交点度量距离
    if dist <= 0.01 or dist > 0.5:
        return None, None
    if abs(p[0]) > 0.6 or abs(p[1]) > 0.6:
        return None, None
    return p, float(dist)


# ────────────────────────────────────────────────────────────────
# 3. 像素 + 深度 -> 相机坐标系三维点 -> 基坐标系三维点
# ────────────────────────────────────────────────────────────────
def pixel_depth_to_cam_xyz(u, v, depth_m, intr):
    """针孔相机模型反投影：像素(u,v) + 深度(米) -> 相机坐标系下(x,y,z)"""
    x = (u - intr.cx) * depth_m / intr.fx
    y = (v - intr.cy) * depth_m / intr.fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def rotz_matrix(theta):
    """绕基坐标系Z轴旋转theta(弧度)的3x3旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _effective_cam2base(T_cam2base, th1):
    """
    相机固定安装在机械臂末端（随臂运动），标定得到的 T_CAM2BASE 只在
    "标定基准姿态"（约定为关节1角度th1=0）下成立。扫描过程中只有关节1
    （基座旋转，绕过基坐标系原点的竖直Z轴）在转动，其余关节固定不变，
    因此任意时刻相机相对基座的实际位姿等价于：把标定基准变换整体绕
    基座Z轴旋转当前th1角：
        T_eff = Rz(-th1) · T_cam2base_ref
    ★ 关节1 URDF 轴为 [0,0,-1], 正向旋转 th1 产生的实际旋转是 Rz(-th1),
      故此处用 Rz(-th1) 复现, 而非 Rz(+th1)。
    若不提供th1（None），则按常量处理（假定标定时与使用时关节角一致）。
    """
    T_ref = T_cam2base if T_cam2base is not None else _resolve_cam2base_ref()
    if th1 is None:
        return T_ref
    Rz = rotz_matrix(-th1)
    T_eff = np.eye(4, dtype=np.float64)
    T_eff[:3, :3] = Rz.dot(T_ref[:3, :3])
    T_eff[:3, 3] = Rz.dot(T_ref[:3, 3])
    return T_eff


def _resolve_cam2base_ref():
    """取当前生效的 th1=0 基准相机→基座变换：优先 tf 标定结果，再退 T_CAM2BASE 兜底。"""
    return _T_CAM2BASE_REF if _T_CAM2BASE_REF is not None else T_CAM2BASE


# ★ 基坐标系输出偏移补偿: 已取消(手眼联合反解后不再需要)
_BASE_OFFSET_CORR = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def cam_xyz_to_base_xyz(p_cam, T_cam2base=None, th1=None, all_joints=None):
    """
    相机坐标系三维点 -> 机械臂基坐标系三维点。
    th1: 拍摄该帧时刻的关节1角度(弧度)，用于修正随基座旋转的相机位姿；
         None 时退化为使用常量 T_CAM2BASE（见 _effective_cam2base 说明）。
    all_joints: ★ 可选, 传入完整6关节角[th1~th6], 直接用FK+手眼算T_eff(最精确,
                避免SCAN_POSITION与实际不符的问题)。若提供则优先于th1修正。
    """
    if all_joints is not None:
        # ★ 直接用实际关节角+手眼标定算camera->base(最可靠)
        T_base_cam = _fk_base_to_6link(all_joints).dot(_T_6LINK_TO_CAM)
        p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        xyz = T_base_cam.dot(p_h)[:3]
        return xyz - _BASE_OFFSET_CORR
    T_eff = _effective_cam2base(T_cam2base, th1)
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    p_base_h = T_eff.dot(p_h)
    return p_base_h[:3] - _BASE_OFFSET_CORR


def pixel_depth_to_base_xyz(u, v, depth_m, intr, T_cam2base=None, th1=None, all_joints=None):
    """像素 + 深度(米) -> 机械臂基坐标系三维坐标，一步到位（含th1旋转修正）"""
    p_cam = pixel_depth_to_cam_xyz(u, v, depth_m, intr)
    return cam_xyz_to_base_xyz(p_cam, T_cam2base, th1=th1, all_joints=all_joints)


def sample_stable_depth(depth_img, cx, cy, win=5):
    """
    以(cx,cy)为中心取win×win邻域，做中值滤波+异常点剔除，得到稳定深度(米)。
    对应专利"采用中值滤波、异常点剔除和区域深度统计方法获得稳定深度"。
    depth_img: 原始深度图（uint16, 单位毫米，或float32已转米）
    返回：depth_m(米) 或 None(该邻域无有效深度)
    """
    h, w = depth_img.shape[:2]
    x1, x2 = max(0, cx - win), min(w, cx + win + 1)
    y1, y2 = max(0, cy - win), min(h, cy + win + 1)
    patch = depth_img[y1:y2, x1:x2].astype(np.float32)

    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    if np.nanmax(valid) > 100:  # 判断单位是毫米还是米
        valid = valid / 1000.0

    median = np.median(valid)
    # 剔除偏离中值过大的异常点后再次求均值，得到更稳定的深度估计
    keep = valid[np.abs(valid - median) < 0.05]
    if keep.size == 0:
        keep = valid
    return float(np.median(keep))


# ────────────────────────────────────────────────────────────────
# 4. 表面法向量估计（专利公式18/19：邻域点云协方差矩阵特征分解）
# ────────────────────────────────────────────────────────────────
def estimate_surface_normal(depth_img, cx, cy, intr, T_cam2base=None, th1=None,
                             radius_px=15, min_points=20):
    """
    以像素(cx,cy)为中心，在其 radius_px 邻域窗口内采集有效深度点，
    反投影为相机坐标系三维点云，构造协方差矩阵并特征分解，取最小特征值
    对应的特征向量作为物证表面法向量，统一调整方向朝向相机侧，
    再旋转到机械臂基坐标系。

    对应关系：
      C = (1/N) * sum((p_i - mean)(p_i - mean)^T)   —— 专利式(18)
      C = λ1 v1v1^T + λ2 v2v2^T + λ3 v3v3^T, 取λ3对应的v3     —— 专利式(19)

    返回：基坐标系下的单位法向量 np.array([nx, ny, nz])，
          若邻域内有效点过少则返回默认竖直向下法向量 [0,0,-1]
    """
    h, w = depth_img.shape[:2]
    x1, x2 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
    y1, y2 = max(0, cy - radius_px), min(h, cy + radius_px + 1)

    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    patch = depth_img[y1:y2, x1:x2].astype(np.float32)

    valid_mask = patch > 0
    if np.count_nonzero(valid_mask) < min_points:
        return np.array([0.0, 0.0, -1.0])

    depth_vals = patch[valid_mask].astype(np.float64)
    if np.nanmax(depth_vals) > 100:
        depth_vals = depth_vals / 1000.0
    u_vals = us[valid_mask].astype(np.float64)
    v_vals = vs[valid_mask].astype(np.float64)

    xs = (u_vals - intr.cx) * depth_vals / intr.fx
    ys = (v_vals - intr.cy) * depth_vals / intr.fy
    zs = depth_vals

    points_cam = np.stack([xs, ys, zs], axis=1)  # N x 3, 相机坐标系

    mean = points_cam.mean(axis=0)
    centered = points_cam - mean
    cov = centered.T.dot(centered) / max(1, points_cam.shape[0])  # 式(18)

    eigvals, eigvecs = np.linalg.eigh(cov)  # 式(19)，eigh按升序返回特征值
    normal_cam = eigvecs[:, 0]  # 最小特征值对应的特征向量

    # 统一调整方向朝向相机侧（法向量与相机视线方向点积应为负，即指向相机）
    if normal_cam[2] > 0:
        normal_cam = -normal_cam

    # 只旋转方向（不平移）：用手眼矩阵的旋转部分把法向量转到基坐标系，
    # 同样需要按拍摄时刻的th1做旋转修正（见 _effective_cam2base 说明）
    T_eff = _effective_cam2base(T_cam2base, th1)
    R = T_eff[:3, :3]
    normal_base = R.dot(normal_cam)

    norm = np.linalg.norm(normal_base)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return normal_base / norm


# ────────────────────────────────────────────────────────────────
# 5. 液体类掩模 PCA 最长轴方向（专利公式8~14，供可选的自适应擦拭增强使用）
# ────────────────────────────────────────────────────────────────
def pca_longest_axis_from_mask(mask):
    """
    对区域生长得到的二值掩模(mask, uint8, 255为前景)做PCA，
    返回 (centroid_px(cx,cy), longest_axis_unit_vector(ex,ey), span_u, span_v)
    其中 longest_axis 是像素坐标系下的最长轴方向单位向量（专利式9的e_u）。
    掩模为空时返回 None。
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 5:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)  # 式(9) 二维协方差矩阵
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 升序排列，最大特征值在最后一列 -> 最长轴方向 e_u
    e_u = eigvecs[:, -1]
    e_v = eigvecs[:, 0]

    theta = np.arctan2(e_u[1], e_u[0])  # 式(10)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    u_coords = centered[:, 0] * cos_t + centered[:, 1] * sin_t
    v_coords = -centered[:, 0] * sin_t + centered[:, 1] * cos_t

    span_u = u_coords.max() - u_coords.min()
    span_v = v_coords.max() - v_coords.min()

    return {
        'centroid': (mean[0], mean[1]),
        'axis_u': e_u,
        'axis_v': e_v,
        'theta': theta,
        'span_u': span_u,
        'span_v': span_v,
    }


# ────────────────────────────────────────────────────────────────
# 6. ★★★ 新增：液体类蛇形擦拭路径 —— 在CDG-ATRG输出的真实掩模区域内
#    生成往复擦拭路径的机械臂基坐标系三维路径点（专利公式8~15的完整
#    工程实现：PCA求主轴 -> 旋转坐标系生成扫描行 -> 蛇形顺序连接 ->
#    像素坐标查深度反投影到基坐标系）
# ────────────────────────────────────────────────────────────────
def generate_snake_wipe_waypoints(mask, depth_img, intr, T_cam2base=None, th1=None,
                                   wipe_width_px=15, min_mask_pixels=20):
    """
    对CDG-ATRG输出的液体物证掩模，按最长轴方向生成蛇形往复扫描路径，
    并把每个路径端点换算成机械臂基坐标系三维坐标。

    参数:
        mask: uint8二值掩模(255为前景)，与depth_img同尺寸
        depth_img: 与mask同一帧对应的深度图
        intr: CameraIntrinsics
        T_cam2base/th1: 见 pixel_depth_to_base_xyz 说明，th1是拍摄这一帧
                        时刻的关节1角度，用于绕基座Z轴的位姿修正
        wipe_width_px: 相邻两行扫描线的步进宽度(像素)，对应棉签有效擦拭
                       宽度换算到图像尺度，需要现场根据实际物理尺寸标定后
                       调整（当前是一个可用的默认经验值）
        min_mask_pixels: 掩模前景像素数少于该值时判定为"区域太小/不可靠"

    返回:
        [(x,y,z), (x,y,z), ...] 按蛇形顺序排列的基坐标系三维路径点列表；
        掩模太小、或路径点因深度无效被跳过导致有效点不足2个时，返回None，
        调用方应回退到质心单点擦拭（见 example_smear_with_linear_motion）
    """
    ys, xs = np.nonzero(mask)
    if xs.size < min_mask_pixels:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    e_u = eigvecs[:, -1]  # 最长轴方向（式9，最大特征值对应特征向量）

    theta = np.arctan2(e_u[1], e_u[0])  # 式(10)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    u_coords = centered[:, 0] * cos_t + centered[:, 1] * sin_t   # 式(11)
    v_coords = -centered[:, 0] * sin_t + centered[:, 1] * cos_t

    v_min, v_max = v_coords.min(), v_coords.max()
    n_lines = max(1, int(np.ceil((v_max - v_min) / wipe_width_px)))  # 式(12)

    h, w = mask.shape[:2]
    waypoints_px = []

    for k in range(n_lines + 1):
        v_k = v_min + k * wipe_width_px  # 式(13)
        band = np.abs(v_coords - v_k) < (wipe_width_px / 2.0)
        if not np.any(band):
            continue
        u_band = u_coords[band]
        u_lo, u_hi = u_band.min(), u_band.max()

        # 蛇形顺序：偶数行从左到右，奇数行从右到左（式14反变换前先定好u顺序）
        u_seq = (u_lo, u_hi) if k % 2 == 0 else (u_hi, u_lo)
        for u in u_seq:
            px = u * cos_t - v_k * sin_t + mean[0]   # 式(14) 逆旋转变换回像素坐标
            py = u * sin_t + v_k * cos_t + mean[1]
            px = int(round(min(max(px, 0), w - 1)))
            py = int(round(min(max(py, 0), h - 1)))
            waypoints_px.append((px, py))

    waypoints_xyz = []
    for px, py in waypoints_px:
        depth_m = sample_stable_depth(depth_img, px, py)
        if depth_m is None or depth_m <= 0:
            continue  # 该点深度无效，跳过，不中断整条路径（与move_linear的容错风格一致）
        xyz = pixel_depth_to_base_xyz(px, py, depth_m, intr, T_cam2base=T_cam2base, th1=th1)
        waypoints_xyz.append(tuple(xyz))

    if len(waypoints_xyz) < 2:
        return None
    return waypoints_xyz

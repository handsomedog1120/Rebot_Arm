#!/usr/bin/env python3
import serial.tools.list_ports
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import rospy
import numpy as np
from numpy import sin, cos, pi, arctan2, arccos, sqrt
from scservo_sdk import scscl as BaseSCSCL
from scservo_sdk import *
from PyQt5.QtCore import (QProcessEnvironment, Qt, QProcess, QTimer, QThread, pyqtSignal)

env = QProcessEnvironment()
from sensor_msgs.msg import JointState, Image, CameraInfo
from functions import workSpace, inv_k, inv_k_best
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox, QTextEdit,
                             QMessageBox, QSplitter, QFormLayout, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
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
JOINT_NAMES = [
    'axis_joint_1',  # 基座旋转
    'axis_joint_2',  # 肩部俯仰
    'axis_joint_3',  # 肘部俯仰
    'axis_joint_4',  # 腕部旋转
    'axis_joint_5',  # 腕部俯仰
    'axis_joint_6'   # 末端旋转
]

# 运动参数
MOVE_DURATION = 5.0       # 单次移动时间(秒)
PAUSE_DURATION = 2.0      # 暂停时间(秒)
INITIAL_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始位置
SAFE_HEIGHT = 0.13         # 安全高度H(米)
SERVO_SPEED = 150         # 舵机速度（SCS协议，范围50-500，约220 RPM）

# ★ 修改：扫描姿态定义（末端俯视地面）
#   关节角度设置：基座水平，大臂小臂使末端指向地面，腕部调整使末端垂直向下
#   典型姿态：th1=0, th2=0.5, th3=-1.5, th4=0, th5=0, th6=0
SCAN_POSITION = [0.0, 0.0, 1.0, -0.7, 1.5, 0.0]

# 扫描参数
SCAN_SWEEP_ANGLE = 4.0    # 扫描摆动角度范围（弧度，约115度）
# ★ 注：扫描摆动幅度 ±SCAN_SWEEP_ANGLE/2 = ±2.0 rad，小于 functions.py 里
#   inv_k() 的关节限位 (-2.79, 2.79)，因此扫描过程天然满足限位要求，
#   自动采集流程直接复用 start_scan()/update_scan_movement()，不重新设计。

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
SCAN_HALF_CIRCLE_TH1_END = 2.7      # 半圈扫描结束关节1角度(rad)
SCAN_ROTATE_DURATION = 15.0         # 半圈旋转总时长(秒)，期间持续检测
POST_SCAN_PAUSE = 5.0               # 旋转到位、机械臂停止后的停顿时间(秒)

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
# 三类物证（涂抹后的棉签、粘取后的粘取棒、抓取到的固体物证）动作完成后，
# 不再放回原来的工具架位置（避免污染工具架、混淆待用工具与已沾染物证的
# 工具），而是统一转移到这个回收位释放，供后续装袋/编号等物证管理流程
# 使用。
# ★ 这是一个占位坐标，需要现场根据实际证物台/回收框位置量取后修改；
#   取值需同时满足 functions.workSpace() 的硬性限制(X/Y/Z均在±0.40m内)，
#   并尽量落在UI"工作空间限制"面板标注的常用范围内(X:0.15~0.40,
#   Y:-0.10~0.30, Z:0.10~0.35，仅供参考，非代码强制限制)。
EVIDENCE_RECOVERY_POINT = [0.30, -0.05, 0.15]  # TODO: 现场标定后替换为真实回收位坐标


class EnhancedSCSCL(BaseSCSCL):
    """扩展舵机控制类，添加速度控制方法"""

    def WriteSpeed(self, scs_id, speed, acc=0):
        """
        SCS速度控制协议扩展
        :param speed: 速度值(单位: 0.074°/s)
        :param acc: 加速度(0-255)
        """
        txpacket = [
            0xFF, 0xFF, scs_id, 0x07, 0x03, 0x20,
            speed & 0xFF, (speed >> 8) & 0xFF,
            acc & 0xFF, (acc >> 8) & 0xFF
        ]
        checksum = (~sum(txpacket[2:])) & 0xFF
        txpacket.append(checksum)
        self.portHandler.write(txpacket)
        return self.read_response()


class ServoScanThread(QThread):
    found_servo = pyqtSignal(int, str)

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name
        self.running = True

    def run(self):
        """线程主循环：扫描 1-253 号舵机"""
        try:
            portHandler = PortHandler(self.port_name)
            if portHandler.openPort() and portHandler.setBaudRate(115200):
                packetHandler = scscl(portHandler)
                for scs_id in range(1, 254):
                    if not self.running:
                        break
                    model_number, comm_result, _ = packetHandler.ping(scs_id)
                    if comm_result == COMM_SUCCESS:
                        self.found_servo.emit(scs_id, f"ID:{scs_id} 型号:{model_number}")
                portHandler.closePort()
        except Exception as e:
            print(f"扫描异常: {str(e)}")

    def stop(self):
        """安全停止扫描线程"""
        self.running = False
        self.wait()


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
        while self._running:
            if not self._active:
                self.msleep(20)  # 非扫描阶段空闲小睡，避免忙等占满CPU
                continue

            color_bgr, depth_img, th1 = self.frame_provider()
            if color_bgr is None:
                self.msleep(20)
                continue

            det = self.detector.detect(color_bgr, depth_img)
            if det is not None and self._running:
                self.detection_ready.emit(det, depth_img, th1)
            # 不加额外sleep：检测节奏完全由detector.run()本身的推理耗时决定，
            # 跑完一帧立刻取最新一帧继续，实现"连续检测"

    def stop(self):
        self._running = False
        self.wait()


class ArmControllerUI(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.current_action = None
        self.action_buttons = []
        self.servo_controller = None   # scscl 实例
        self.port_handler = None       # ★ 统一持有 PortHandler，避免端口冲突
        self.scan_thread = None

        self.setWindowTitle("刑侦现场物证采集系统")
        self.setGeometry(100, 100, 1920, 1080)

        # 科技风配色方案
        self.COLOR_BG = QColor(20, 30, 50)
        self.COLOR_PANEL = QColor(40, 50, 70)
        self.COLOR_ACCENT = QColor(0, 180, 255)
        self.COLOR_TEXT = QColor(255, 255, 255)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                color: {self.COLOR_TEXT.name()};
                background-color: {self.COLOR_BG.name()};
            }}
            QGroupBox {{
                border: 2px solid {self.COLOR_ACCENT.name()};
                border-radius: 8px;
                margin-top: 1em;
                padding: 15px;
                background-color: {self.COLOR_PANEL.name()};
            }}
            QGroupBox::title {{
                color: {self.COLOR_ACCENT.name()};
                subcontrol-origin: margin;
                left: 10px;
            }}
            QLabel, QLineEdit, QTextEdit, QPushButton {{
                color: {self.COLOR_TEXT.name()};
            }}
            QLineEdit {{
                border: 1px solid {self.COLOR_ACCENT.name()};
                border-radius: 4px;
                padding: 8px;
                background-color: {self.COLOR_PANEL.name()};
            }}
            QPushButton {{
                border: 2px solid {self.COLOR_ACCENT.name()};
                border-radius: 5px;
                padding: 8px 15px;
                background-color: {self.COLOR_PANEL.name()};
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {self.COLOR_ACCENT.name()}20;
            }}
            QTextEdit {{
                background-color: {self.COLOR_PANEL.name()};
                border: 1px solid {self.COLOR_ACCENT.name()};
                border-radius: 5px;
                padding: 10px;
            }}
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("GLUON CONTROL SYSTEM")
        title.setStyleSheet(f"""
            font-size: 32px;
            font-weight: bold;
            color: {self.COLOR_ACCENT.name()};
            padding: 15px;
            border-bottom: 3px solid {self.COLOR_ACCENT.name()};
        """)
        self.statusBar().setStyleSheet(f"background-color: {self.COLOR_PANEL.name()};")

        self.launch_process = QProcess()
        self.launch_process.readyReadStandardOutput.connect(self.handle_launch_output)
        self.launch_process.readyReadStandardError.connect(self.handle_launch_error)
        self.launch_process.finished.connect(self.handle_launch_finished)

        self.camera_processes = []
        self.init_ui()
        
        # ★ 重要：在 show_current_status() 之前初始化所有状态变量
        # 机械臂移动相关变量
        self.is_moving = False
        self.move_start_time = 0
        self.move_duration = 0
        self.move_start_pos = []
        self.move_end_pos = []
        self.current_move_index = 0
        self.move_sequence = []
        self.is_pausing = False
        self.pause_start_time = 0
        
        # ★ 夹爪控制相关变量
        self.is_gripper_waiting = False
        self.gripper_wait_start = 0
        self.gripper_wait_duration = 0

        # ★ 扫描状态变量（必须在 show_current_status 之前初始化）
        self.is_scanning = False
        self.scan_direction = 1  # 1: 正向摆动, -1: 反向摆动
        self.scan_start_angle = 0.0
        self.scan_current_angle = 0.0
        self.scan_target_angle = 0.0
        self.scan_start_time = 0.0
        self._last_scan_status_time = 0
        self.scan_base_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ──────────────────────────────────────────────────────────────
        # ★★★ 新增：自动物证采集流程状态变量 ★★★
        # ──────────────────────────────────────────────────────────────
        self.auto_mode = False          # 是否处于自动采集模式
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
        self.raw_color_image = None      # 最近一帧原始彩色图 (numpy, BGR)
        self.raw_depth_image = None      # 最近一帧原始深度图 (numpy, uint16/mm 或 float32/m)
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

        self.init_servo_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # 舵机 UI 初始化
    # ──────────────────────────────────────────────────────────────────────────
    def init_servo_ui(self):
        servo_group = QGroupBox("舵机夹爪控制")
        servo_layout = QVBoxLayout()

        port_group = QGroupBox("通信配置")
        port_layout = QFormLayout()

        self.port_combo = QComboBox()
        self.btn_refresh = QPushButton("刷新端口")
        self.btn_refresh.clicked.connect(self.refresh_ports)

        self.btn_scan = QPushButton("扫描舵机")
        self.btn_scan.clicked.connect(self.start_servo_scan)
        self.servo_list = QComboBox()

        port_layout.addRow("通信端口:", self.port_combo)
        port_layout.addRow(self.btn_refresh)
        port_layout.addRow("扫描舵机:", self.servo_list)
        port_layout.addRow(self.btn_scan)
        port_group.setLayout(port_layout)

        control_group = QGroupBox("舵机控制")
        control_layout = QFormLayout()

        self.angle_input = QLineEdit("300")
        self.btn_set_angle = QPushButton("设置角度")
        self.btn_set_angle.clicked.connect(self.set_servo_angle)

        control_layout.addRow("目标角度:", self.angle_input)
        control_layout.addRow(self.btn_set_angle)

        gripper_group = QGroupBox("夹爪控制")
        gripper_layout = QHBoxLayout()

        self.btn_open = QPushButton("打开夹爪 (300)")
        self.btn_open.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_open.clicked.connect(lambda: self._send_gripper(300))

        self.btn_close = QPushButton("闭合夹爪 (500)")
        self.btn_close.setStyleSheet("background-color: #F44336; color: white;")
        self.btn_close.clicked.connect(lambda: self._send_gripper(500))

        gripper_layout.addWidget(self.btn_open)
        gripper_layout.addWidget(self.btn_close)
        gripper_group.setLayout(gripper_layout)

        control_layout.addRow(gripper_group)
        control_group.setLayout(control_layout)

        servo_layout.addWidget(port_group)
        servo_layout.addWidget(control_group)
        servo_group.setLayout(servo_layout)

        left_panel = self.centralWidget().findChild(QWidget)
        if left_panel and left_panel.layout():
            left_panel.layout().insertWidget(2, servo_group)
        else:
            self.centralWidget().layout().addWidget(servo_group)

        # 初始化时刷新端口列表
        self.refresh_ports()

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：统一端口管理，安全复用连接
    # ──────────────────────────────────────────────────────────────────────────
    def init_servo_connection(self):
        """
        统一管理 PortHandler。
        - 端口名未变且端口已打开 → 直接复用，不重复开端口
        - 否则关闭旧句柄，等扫描线程退出后再重新打开
        """
        port_name = self.port_combo.currentText()
        if not port_name:
            raise ConnectionError("请先选择通信端口")

        # 复用判断：端口句柄存在、已打开、端口名一致
        if (self.port_handler is not None
                and self.port_handler.is_open
                and getattr(self.port_handler, 'port_name', None) == port_name):
            if self.servo_controller is None:
                self.servo_controller = scscl(self.port_handler)
            return

        # 关闭旧句柄
        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            self.port_handler = None
            self.servo_controller = None

        # 等待扫描线程彻底释放端口
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread = None
        time.sleep(0.3)

        portHandler = PortHandler(port_name)
        for attempt in range(3):
            if portHandler.openPort() and portHandler.setBaudRate(115200):
                self.port_handler = portHandler
                self.servo_controller = scscl(portHandler)
                return
            time.sleep(0.2)

        raise ConnectionError(f"无法打开端口 {port_name}，请确认端口未被占用")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：set_servo_angle — 修正速度参数，区分通信/舵机错误
    # ──────────────────────────────────────────────────────────────────────────
    def set_servo_angle(self):
        try:
            angle = int(self.angle_input.text())
            if not 0 <= angle <= 4095:
                raise ValueError("角度值需在 0–4095 之间")

            if self.servo_list.currentIndex() < 0:
                raise ValueError("请先扫描并选择舵机")

            scs_id = self.servo_list.currentData()
            self.init_servo_connection()

            # ★ 修复：speed 使用合理固定值 SERVO_SPEED(300)
            #    原来 int(100/0.074)≈1351 超出协议范围，舵机会忽略指令
            scs_comm_result, scs_error = self.servo_controller.WritePos(
                scs_id, angle, 0, SERVO_SPEED
            )

            if scs_comm_result == COMM_SUCCESS:
                if scs_error != 0:
                    err_msg = self.servo_controller.getRxPacketError(scs_error)
                    # 通信成功但舵机报错（过载/过温等），用状态栏而非弹窗
                    self.statusBar().showMessage(f"⚠️ 舵机错误: {err_msg}")
                else:
                    self.statusBar().showMessage(
                        f"✅ 舵机 ID={scs_id} 角度已设为 {angle}"
                    )
            else:
                tx_result = self.servo_controller.getTxRxResult(scs_comm_result)
                raise RuntimeError(f"通信失败: {tx_result}")

        except Exception as e:
            QMessageBox.critical(self, "舵机错误", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：夹爪控制 — 移除 time.sleep，改为非阻塞
    # ──────────────────────────────────────────────────────────────────────────
    def _send_gripper(self, angle, wait_time=0.8):
        """
        直接向当前选中舵机发送角度指令（非阻塞）。
        供夹爪按钮和 set_gripper_angle 共同调用。
        """
        try:
            if self.servo_list.currentIndex() < 0:
                self.statusBar().showMessage("⚠️ 未选择夹爪舵机，跳过")
                return False

            self.init_servo_connection()
            scs_id = self.servo_list.currentData()

            scs_comm_result, scs_error = self.servo_controller.WritePos(
                scs_id, angle, 0, SERVO_SPEED
            )

            if scs_comm_result == COMM_SUCCESS and scs_error == 0:
                label = "打开" if angle <= 300 else "闭合"
                self.statusBar().showMessage(f"夹爪{label} ({angle}) ✅")
                return True
            else:
                self.statusBar().showMessage(
                    f"⚠️ 夹爪指令失败 comm={scs_comm_result} err={scs_error}"
                )
                return False
        except Exception as e:
            self.statusBar().showMessage(f"夹爪异常: {e}")
            return False

    def set_gripper_angle(self, angle):
        """
        供 execute_action 等逻辑调用。
        不再通过 angle_input 中转，避免 UI 副作用。
        """
        self._send_gripper(angle)

    # ──────────────────────────────────────────────────────────────────────────
    # 端口刷新
    # ──────────────────────────────────────────────────────────────────────────
    def refresh_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：扫描前彻底关闭 servo_controller 持有的端口
    # ──────────────────────────────────────────────────────────────────────────
    def start_servo_scan(self):
        self.servo_list.clear()
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "警告", "请先选择通信端口!")
            return

        # 停止旧扫描线程
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread = None

        # ★ 关键：关闭 servo_controller 占用的端口，防止端口冲突
        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            self.port_handler = None
            self.servo_controller = None

        time.sleep(0.2)

        self.scan_thread = ServoScanThread(port)
        self.scan_thread.found_servo.connect(self.update_servo_list)
        self.scan_thread.start()

    def update_servo_list(self, scs_id, info):
        self.servo_list.addItem(info, scs_id)

    # ──────────────────────────────────────────────────────────────────────────
    # 定时更新
    # ──────────────────────────────────────────────────────────────────────────
    def update_all(self):
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
        # 处理扫描模式
        if self.is_scanning:
            self.update_scan_movement()
            return
        
        # ★ 新增：处理夹爪等待状态
        if self.is_gripper_waiting:
            current_time = rospy.get_time()
            if current_time - self.gripper_wait_start >= self.gripper_wait_duration:
                self.is_gripper_waiting = False
                self.statusBar().showMessage(f"夹爪动作完成，继续执行...")
            else:
                return  # 仍在等待中，直接返回
        
        if not self.is_moving and not self.is_pausing and not self.is_gripper_waiting:
            if self.current_move_index < len(self.move_sequence):
                move = self.move_sequence[self.current_move_index]
                
                # ★ 检查是否有夹爪控制指令
                if 'gripper' in move:
                    # 执行夹爪控制（发送指令）
                    success = self._send_gripper(move['gripper'])
                    if success:
                        # ★ 非阻塞等待：设置等待标志，不阻塞主线程
                        self.is_gripper_waiting = True
                        self.gripper_wait_start = rospy.get_time()
                        self.gripper_wait_duration = move.get('gripper_wait', 0.8)
                        self.statusBar().showMessage(f"夹爪动作执行中，等待 {self.gripper_wait_duration}s...")
                    self.current_move_index += 1
                    return
                else:
                    self.start_arm_movement(move['start'], move['end'], move['duration'])
                    self.current_move_index += 1
                    return

        if self.is_moving:
            current_time = rospy.get_time()
            elapsed = current_time - self.move_start_time
            fraction = min(elapsed / self.move_duration, 1.0)

            current_pos = interpolate(self.move_start_pos, self.move_end_pos, fraction)
            pub2ros_pos(self.controller.publisher, current_pos)

            self.controller.joint_pos = current_pos
            self.controller.q6_value = current_pos[5]

            if fraction >= 1.0:
                self.is_moving = False
                self.is_pausing = True
                self.pause_start_time = current_time
                self.show_current_status()

        elif self.is_pausing:
            current_time = rospy.get_time()
            if current_time - self.pause_start_time >= PAUSE_DURATION:
                self.is_pausing = False
                if self.current_move_index >= len(self.move_sequence):
                    self.move_sequence = []
                    self.current_move_index = 0
                    self.current_action = None
                    self.enable_action_buttons(True)
                    self.show_current_status()

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 新增：扫描运动更新函数
    # ──────────────────────────────────────────────────────────────────────────
    def update_scan_movement(self):
        """更新扫描运动：绕关节1来回摆动"""
        current_time = rospy.get_time()
        dt = 0.03  # 更新间隔（与定时器匹配）
        
        # 计算当前扫描角度（使用正弦波实现平滑来回摆动）
        # 频率：每秒完成一次完整来回
        frequency = 0.05  # Hz，每秒半个周期
        elapsed = current_time - self.scan_start_time
        
        # 使用正弦波生成摆动角度，范围 [-SCAN_SWEEP_ANGLE/2, SCAN_SWEEP_ANGLE/2]
        angle_offset = (SCAN_SWEEP_ANGLE / 2) * sin(2 * pi * frequency * elapsed)
        
        # 更新关节1角度
        new_pos = self.scan_base_position.copy()
        new_pos[0] = self.scan_base_position[0] + angle_offset
        
        # 发布位置
        pub2ros_pos(self.controller.publisher, new_pos)
        self.controller.joint_pos = new_pos
        self.controller.q6_value = new_pos[5]
        
        # 更新状态显示（每秒更新一次状态栏）
        if int(elapsed) % 1 == 0 and not hasattr(self, '_last_scan_status_time') or \
           int(elapsed) != self._last_scan_status_time:
            self._last_scan_status_time = int(elapsed)
            self.statusBar().showMessage(
                f"🔍 扫描中... 角度: {angle_offset:.3f} rad, 时间: {elapsed:.1f}s"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # UI 初始化
    # ──────────────────────────────────────────────────────────────────────────
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        self.statusBar().showMessage("系统就绪")
    
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)

        control_group = QGroupBox("机械臂控制")
        control_layout = QVBoxLayout()

        # 笛卡尔空间控制
        cartesian_group = QGroupBox("笛卡尔空间控制")
        cartesian_layout = QFormLayout()

        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()
        self.j6_input = QLineEdit()

        cartesian_layout.addRow("X 坐标 (米):", self.x_input)
        cartesian_layout.addRow("Y 坐标 (米):", self.y_input)
        cartesian_layout.addRow("Z 坐标 (米):", self.z_input)
        cartesian_layout.addRow("关节6角度 (弧度):", self.j6_input)

        # ★ 新增：表面法向量输入
        self.nx_input = QLineEdit("0")
        self.ny_input = QLineEdit("0")
        self.nz_input = QLineEdit("-1")
        
        # 添加法向量输入行
        normal_layout = QHBoxLayout()
        normal_layout.addWidget(QLabel("nx:"))
        normal_layout.addWidget(self.nx_input)
        normal_layout.addWidget(QLabel("ny:"))
        normal_layout.addWidget(self.ny_input)
        normal_layout.addWidget(QLabel("nz:"))
        normal_layout.addWidget(self.nz_input)
        
        cartesian_layout.addRow("表面法向量:", normal_layout)
        
        # 添加法向量显示标签（显示当前四元数）
        self.normal_status_label = QLabel("当前姿态: [1.0, 0.0, 0.0, 0.0]")
        self.normal_status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        cartesian_layout.addRow(self.normal_status_label)

        move_btn = QPushButton("移动到位置")
        move_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        move_btn.clicked.connect(self.move_to_position)
        cartesian_layout.addRow(move_btn)


        # ★ 修改：开始扫描按钮（替换原来的开始操作）
        start_scan_btn = QPushButton("开始扫描")
        start_scan_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        start_scan_btn.clicked.connect(self.start_scan)
        cartesian_layout.addRow(start_scan_btn)

        home_btn = QPushButton("返回初始位置")
        home_btn.setStyleSheet("background-color: #FF9800; color: white;")
        home_btn.clicked.connect(self.return_to_zero)
        cartesian_layout.addRow(home_btn)

        # ★★★ 新增：自动物证采集流程按钮 ★★★
        # 打通 初始化->扫描->CenterNet检测->标定/深度反投影->按类别执行 的整条流程
        auto_start_btn = QPushButton("自动识别")
        auto_start_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        auto_start_btn.clicked.connect(self.start_auto_pipeline)
        cartesian_layout.addRow(auto_start_btn)

        auto_stop_btn = QPushButton("停止自动采集")
        auto_stop_btn.setStyleSheet("background-color: #607D8B; color: white;")
        auto_stop_btn.clicked.connect(self.stop_auto_pipeline)
        cartesian_layout.addRow(auto_stop_btn)

        cartesian_group.setLayout(cartesian_layout)

        # 预设动作控制
        action_group = QGroupBox("预设动作控制")
        action_layout = QVBoxLayout()

        height_layout = QHBoxLayout()
        height_label = QLabel("安全高度H (米):")
        self.height_input = QLineEdit(str(SAFE_HEIGHT))
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)
        action_layout.addLayout(height_layout)

        btn_layout1 = QHBoxLayout()
        self.action1_btn = QPushButton("执行动作1(涂抹)")
        self.action1_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action1_btn.clicked.connect(lambda: self.execute_action(1))
        btn_layout1.addWidget(self.action1_btn)

        btn_layout2 = QHBoxLayout()
        self.action2_btn = QPushButton("执行动作2(粘取)")
        self.action2_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action2_btn.clicked.connect(lambda: self.execute_action(2))
        btn_layout2.addWidget(self.action2_btn)

        btn_layout3 = QHBoxLayout()
        self.action3_btn = QPushButton("执行动作3(抓取)")
        self.action3_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action3_btn.clicked.connect(lambda: self.execute_action(3))
        btn_layout3.addWidget(self.action3_btn)

        action_layout.addLayout(btn_layout1)
        action_layout.addLayout(btn_layout2)
        action_layout.addLayout(btn_layout3)

        self.action_buttons = [self.action1_btn, self.action2_btn, self.action3_btn]
        action_group.setLayout(action_layout)

        # 关节空间控制
        joint_group = QGroupBox("关节空间控制")
        joint_layout = QVBoxLayout()

        rotate_layout = QHBoxLayout()
        self.rotate_amount = QLineEdit("0.1")

        cw_btn = QPushButton("顺时针旋转")
        cw_btn.setStyleSheet("background-color: #2196F3; color: white;")
        cw_btn.clicked.connect(lambda: self.rotate_j6(float(self.rotate_amount.text())))

        ccw_btn = QPushButton("逆时针旋转")
        ccw_btn.setStyleSheet("background-color: #2196F3; color: white;")
        ccw_btn.clicked.connect(lambda: self.rotate_j6(-float(self.rotate_amount.text())))

        rotate_layout.addWidget(QLabel("旋转角度:"))
        rotate_layout.addWidget(self.rotate_amount)
        rotate_layout.addWidget(cw_btn)
        rotate_layout.addWidget(ccw_btn)
        joint_layout.addLayout(rotate_layout)
        joint_group.setLayout(joint_layout)

        # 系统控制
        system_group = QGroupBox("系统控制")
        system_layout = QHBoxLayout()

        start_btn = QPushButton("启动机械臂")
        start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        start_btn.clicked.connect(self.start_arm)

        stop_btn = QPushButton("紧急停止")
        stop_btn.setStyleSheet("background-color: #F44336; color: white;")
        stop_btn.clicked.connect(self.emergency_stop)

        system_layout.addWidget(start_btn)
        system_layout.addWidget(stop_btn)
        system_group.setLayout(system_layout)

        control_layout.addWidget(cartesian_group)
        control_layout.addWidget(joint_group)
        control_layout.addWidget(action_group)
        control_layout.addWidget(system_group)
        control_group.setLayout(control_layout)

        # 工作空间信息
        ws_group = QGroupBox("工作空间限制")
        ws_layout = QVBoxLayout()
        ws_layout.addWidget(QLabel("X轴范围: 0.15 ~ 0.40 (米)"))
        ws_layout.addWidget(QLabel("Y轴范围: -0.10 ~ 0.30 (米)"))
        ws_layout.addWidget(QLabel("Z轴范围: 0.10 ~ 0.35 (米)"))
        ws_group.setLayout(ws_layout)

        layout.addWidget(control_group)
        layout.addWidget(ws_group)
        layout.addStretch()

        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        camera_group = QGroupBox("相机视图")
        camera_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setText("等待相机图像...")
        self.camera_label.setStyleSheet("background-color: black; color: white;")

        cam_control_layout = QHBoxLayout()

        start_cam_btn = QPushButton("启动相机")
        start_cam_btn.setStyleSheet("background-color: #2196F3; color: white;")
        start_cam_btn.clicked.connect(self.start_camera)

        stop_cam_btn = QPushButton("停止相机")
        stop_cam_btn.setStyleSheet("background-color: #F44336; color: white;")
        stop_cam_btn.clicked.connect(self.stop_camera)

        cam_control_layout.addWidget(start_cam_btn)
        cam_control_layout.addWidget(stop_cam_btn)

        camera_layout.addWidget(self.camera_label)
        camera_layout.addLayout(cam_control_layout)
        camera_group.setLayout(camera_layout)

        # ★★★ 新增：检测信息实时显示框 ★★★
        # 扫描过程中每次相机检测到物证, 都在这里显示:
        #   类别 / 置信度 / 像素坐标 / 深度 / 相机系坐标 / th1 / 转化后基坐标系坐标
        # 便于排查"相机检测到的位置"与"转化给机械臂的坐标"是否正确
        detect_group = QGroupBox("检测信息 (相机检测 -> 转化坐标)")
        detect_layout = QVBoxLayout()
        self.detection_display = QTextEdit()
        self.detection_display.setReadOnly(True)
        self.detection_display.setMinimumHeight(160)
        # ★ 关键: 必须同时覆盖 color 和 background-color, 否则只改背景不改字色,
        #   会继承全局样式表的白色文字 -> 白底白字不可见
        self.detection_display.setStyleSheet(
            "QTextEdit { font-family: 'Consolas','Monospace','Courier New',monospace;"
            " font-size: 12px; color: #00ff88; background-color: #102030;"
            " border: 1px solid #00b4ff; border-radius: 5px; padding: 8px; }")
        detect_layout.addWidget(self.detection_display)
        detect_group.setLayout(detect_layout)

        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMinimumHeight(200)

        status_layout.addWidget(self.status_display)
        status_group.setLayout(status_layout)

        layout.addWidget(camera_group)
        layout.addWidget(detect_group)
        layout.addWidget(status_group)

        return panel

    def update_camera_view(self):
        """完全绕过cv_bridge的图像处理方法"""
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
                    Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)

            except Exception as e:
                self.camera_label.setText(f"图像处理错误: {str(e)}")
                rospy.logerr(f"更新相机视图错误: {str(e)}")

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
        
        # ★ 用FK从实际关节角算末端真实位置(inv_k有~9cm误差, 输入坐标≠实际位置)
        try:
            joints_actual = list(self.controller.joint_pos)
            joints_actual[5] = self.controller.q6_value
            T_link = pcalib._fk_base_to_6link(joints_actual)
            orn_mat = pcalib._quat_to_matrix(quat[0], quat[1], quat[2], quat[3])
            d7 = 0.165
            end_pos = T_link[:3, 3] + T_link[:3, :3].dot(orn_mat[:3, 2]) * d7
            fk_str = "X: {:.4f}\n    Y: {:.4f}\n    Z: {:.4f}".format(*end_pos)
        except:
            fk_str = "N/A"
        
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
    def move_to_position(self):
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

            success, joints = self.controller.move_to_cartesian(x, y, z, q6, normal)

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
                
                self.start_arm_movement(self.controller.joint_pos, joints, MOVE_DURATION)
                QMessageBox.information(self, "成功", 
                    f"机械臂开始移动!\n\n"
                    f"位置: ({x:.3f}, {y:.3f}, {z:.3f})\n"
                    f"法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})\n"
                    f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n\n"
                    f"目标关节角 (rad):\n"
                    f"J1={joints[0]:.4f}, J2={joints[1]:.4f}, J3={joints[2]:.4f}\n"
                    f"J4={joints[3]:.4f}, J5={joints[4]:.4f}, J6={joints[5]:.4f}")
            else:
                QMessageBox.warning(self, "错误", "无法移动到目标位置!")
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请输入有效的数字!\n{str(e)}")

    # ★ 修改：开始扫描方法 - 末端俯视地面，在最大角度内来回旋转
    def start_scan(self):
        """使机械臂进入扫描模式：末端俯视地面，在最大角度内来回旋转"""
        try:
            # 检查是否有正在执行的动作
            if self.current_action is not None:
                QMessageBox.warning(self, "警告", "当前已有动作正在执行!")
                return
            
            # 检查是否已经在扫描中
            if self.is_scanning:
                QMessageBox.warning(self, "警告", "扫描正在进行中!")
                return

            # 获取扫描姿态（末端俯视地面）
            scan_pos = SCAN_POSITION.copy()
            
            # 更新q6值
            self.controller.q6_value = scan_pos[5]
            
            # 保存扫描基础位置（用于摆动）
            self.scan_base_position = scan_pos.copy()
            self.scan_start_time = rospy.get_time()
            self.is_scanning = True
            self._last_scan_status_time = 0
            
            # 移动到扫描姿态
            self.start_arm_movement(self.controller.joint_pos, scan_pos, MOVE_DURATION)
            
            # 显示状态信息
            self.statusBar().showMessage(
                f"🔍 开始扫描: 末端俯视地面，在 ±{SCAN_SWEEP_ANGLE/2:.1f} rad 范围内摆动"
            )
            
            self.status_display.append(
                f"[开始扫描] 姿态: J1={scan_pos[0]}, J2={scan_pos[1]:.3f}, "
                f"J3={scan_pos[2]:.3f}, J4={scan_pos[3]}, J5={scan_pos[4]}, J6={scan_pos[5]}"
            )
            self.status_display.append(
                f"[开始扫描] 摆动范围: ±{SCAN_SWEEP_ANGLE/2:.3f} rad (约 {SCAN_SWEEP_ANGLE/2*180/pi:.1f}°)"
            )
            
            QMessageBox.information(self, "开始扫描", 
                f"机械臂进入扫描模式:\n\n"
                f"末端姿态: 俯视地面\n"
                f"摆动范围: ±{SCAN_SWEEP_ANGLE/2:.3f} rad (约 {SCAN_SWEEP_ANGLE/2*180/pi:.1f}°)\n"
                f"摆动频率: 0.5 Hz (完整来回约2秒)\n\n"
                f"点击「紧急停止」可退出扫描模式")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"开始扫描失败: {str(e)}")

    def stop_scan(self):
        """停止扫描模式"""
        if self.is_scanning:
            self.is_scanning = False
            self.statusBar().showMessage("扫描已停止")
            self.status_display.append("[扫描停止] 扫描模式已退出")
            self.show_current_status()

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
                RAW_COLOR_TOPIC, Image, self.on_raw_color, queue_size=1)
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber(
                RAW_DEPTH_TOPIC, Image, self.on_raw_depth, queue_size=1)
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
        """原始彩色图回调：转换为numpy BGR图像，供CenterNet检测使用"""
        try:
            if msg.encoding == 'rgb8':
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                self.raw_color_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                self.raw_color_image = np.frombuffer(
                    msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            elif msg.encoding == 'mono8':
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                self.raw_color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                rospy.logwarn_throttle(5, "未支持的原始彩色图编码: {}".format(msg.encoding))
        except Exception as e:
            rospy.logerr("原始彩色图转换失败: {}".format(e))

    def on_raw_depth(self, msg):
        """原始深度图回调：转换为numpy深度图（毫米uint16或米float32）"""
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                self.raw_depth_image = np.frombuffer(
                    msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
            elif msg.encoding == '32FC1':
                self.raw_depth_image = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            else:
                rospy.logwarn_throttle(5, "未支持的原始深度图编码: {}".format(msg.encoding))
        except Exception as e:
            rospy.logerr("原始深度图转换失败: {}".format(e))

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

        # 复用已有功能启动机械臂与相机，不改变其行为
        if self.launch_process.state() != QProcess.Running:
            self.start_arm()
        if not self.camera_processes:
            self.start_camera()

        self.auto_mode = True
        self._best_capture = None
        self._enter_auto_state('INIT_HOMING')

        self.status_display.append(
            "[自动采集] 流程已启动：机械臂初始化 -> 关节限位内旋转半圈扫描检测(后台线程连续检测) -> "
            "停止 -> 停顿{:.0f}秒 -> 按检测信息执行动作".format(POST_SCAN_PAUSE))
        self.statusBar().showMessage("🤖 自动采集流程已启动")

    def stop_auto_pipeline(self):
        """
        【停止自动采集】按钮：结束状态机。
        若机械臂正处于半圈扫描/回零等由 start_arm_movement() 驱动的运动中，
        立即清除 is_moving/is_pausing 使其停在当前位置；若正处于
        execute_action() 阻塞执行中，则等该次动作跑完（与原有紧急停止行为一致）。
        后台检测线程不销毁，只是置为非激活状态（空闲小睡），
        下次再次启动自动采集时可以直接复用，不用重新加载模型。
        """
        self.auto_mode = False
        self._enter_auto_state('IDLE')
        self.is_moving = False
        self.is_pausing = False
        self.stop_scan()  # 兼容手动扫描仍在运行的边界情况
        if self.detection_thread is not None:
            self.detection_thread.set_active(False)
        self._best_capture = None
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
        color = self.raw_color_image
        if color is None:
            return None, None, None
        depth = self.raw_depth_image
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
        # ★ 连同depth_img/th1一起记录：液体类的蛇形擦拭路径生成
        # (generate_snake_wipe_waypoints)需要用"确认到这一帧"对应的
        # 深度图+关节1角度，不能用执行动作那一刻(机械臂已回零/已停止)的
        # 当前状态，原理与三维坐标解算要求一致（见compute_target_pose_from_detection注释）
        self._best_capture = (det, xyz, normal, self._pending_depth_img, self._pending_th1)
        self.status_display.append(
            "[自动采集] ✅ 确认物证 {}({}) score={:.2f} -> 基坐标=({:.4f},{:.4f},{:.4f})m "
            "法向=({:.1f},{:.1f},{:.1f}) | 坐标已记录，继续完成本轮扫描".format(
                det.label, det.category, det.score,
                xyz[0], xyz[1], xyz[2],
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
        color_img = self.raw_color_image
        if color_img is not None:
            ch, cw = color_img.shape[:2]
        else:
            ch, cw = 480, 640

        ci = self.camera_intrinsics   # 彩色内参
        di = self.depth_intrinsics    # 深度内参

        depth_m = None
        _map_mode = 'unknown'
        dx, dy = -1, -1

        if dw == cw and dh == ch:
            # 1. 深度已对齐到彩色 (depth_registration=true 生效)
            dx, dy = int(round(cx)), int(round(cy))
            dx = max(0, min(dx, dw - 1))
            dy = max(0, min(dy, dh - 1))
            depth_m = pcalib.sample_stable_depth(depth_img, dx, dy)
            _map_mode = 'aligned(direct)'
        else:
            # 2. 用深度内参+基线补偿映射(硬编码Astra默认值, 不依赖话题)
            dx, dy = self._map_color_to_depth_pixel(cx, cy, depth_img)
            depth_m = pcalib.sample_stable_depth(depth_img, dx, dy)
            _map_mode = 'intrinsics+baseline'

        if depth_m is None or depth_m <= 0:
            self.statusBar().showMessage("⚠️ 目标区域深度无效，跳过本次检测")
            return None

        # ★ 诊断: 打印映射细节(用于排查深度问题)
        print("[深度映射] mode={} 深度图={}×{} 彩色={}×{} "
              "彩色像素=({:.0f},{:.0f}) -> 深度像素=({:.0f},{:.0f}) "
              "depth_m={:.3f}m 深度内参ready={}({:.1f},{:.1f},fx={:.1f},fy={:.1f}) "
              "外参ready={}".format(
                  _map_mode, dw, dh, cw, ch, cx, cy, dx, dy, depth_m,
                  di.ready, di.cx, di.cy, di.fx, di.fy,
                  self._depth_to_color_extrinsics is not None))

        # ★ 首次定位前，用纯 numpy 正运动学 + forensic_ws 手眼标定算
        #   cam2base 基准 (th1=0)，缓存后 _effective_cam2base 按 th1 精确还原该帧相机位姿。
        #   不依赖 ROS tf 库 (conda py3 下 import tf 报 PyInit__tf2 错误)。
        #   用 SCAN_POSITION 的固定 j2~6 + 当前 th1 做 FK (扫描期 j2~6 不变)。
        if not pcalib.cam2base_ref_ready():
            scan_joints = [self.controller.joint_pos[0]] + list(SCAN_POSITION[1:])
            pcalib.set_cam2base_ref_from_fk(scan_joints)

        # ★★★ 坐标转化全链路日志（供排查手眼标定/深度对齐问题）★★★
        intr = self.camera_intrinsics
        p_cam = pcalib.pixel_depth_to_cam_xyz(cx, cy, depth_m, intr)
        # ★ 用实际关节角(而非SCAN_POSITION)算T_eff, 避免姿态不符
        actual_joints = list(self.controller.joint_pos)
        T_fk_eye = pcalib._fk_base_to_6link(actual_joints).dot(pcalib._T_6LINK_TO_CAM)
        print("\n" + "=" * 70)
        print("[坐标转化链路] {}({}) score={:.2f}".format(det.label, det.category, det.score))
        print("  1. 像素坐标(彩色图)  : cx={:.1f}, cy={:.1f}  (图像尺寸 {}×{})".format(
            cx, cy, cw, ch))
        print("  2. 像素映射(深度图)  : dx={}, dy={}  (图像尺寸 {}×{})".format(
            dx, dy, dw, dh))
        print("  3. 深度值            : depth_m={:.4f} m ({:.1f} mm)".format(
            depth_m, depth_m * 1000.0))
        print("  4. 相机内参          : fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}".format(
            intr.fx, intr.fy, intr.cx, intr.cy))
        print("  5. 相机坐标系三维点  : p_cam=({:.4f}, {:.4f}, {:.4f}) m".format(
            p_cam[0], p_cam[1], p_cam[2]))
        print("  6. 实际关节角        : [{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]".format(
            *actual_joints))
        print("  7. SCAN_POSITION     : [{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]".format(
            *SCAN_POSITION))
        print("  8. FK+手眼T_eff(实际关节):")
        print("     平移 = ({:.4f}, {:.4f}, {:.4f})".format(
            T_fk_eye[0, 3], T_fk_eye[1, 3], T_fk_eye[2, 3]))

        # ★ 用实际关节角直接FK+手眼(最可靠, 不依赖SCAN_POSITION/th1修正)
        xyz = pcalib.pixel_depth_to_base_xyz(
            cx, cy, depth_m, self.camera_intrinsics, all_joints=actual_joints)
        print("  9. ★基坐标系三维坐标  : xyz=({:.4f}, {:.4f}, {:.4f}) m".format(
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
                self.return_to_zero()
                self._state_action_triggered = True
                return
            if not self.is_moving and not self.is_pausing:
                self._enter_auto_state('MOVE_TO_SCAN_START')
            return

        if self.auto_state == 'MOVE_TO_SCAN_START':
            if not self._state_action_triggered:
                scan_start_pos = [SCAN_HALF_CIRCLE_TH1_START] + list(SCAN_POSITION[1:])
                self.start_arm_movement(self.controller.joint_pos, scan_start_pos, MOVE_DURATION)
                self._state_action_triggered = True
                return
            if not self.is_moving and not self.is_pausing:
                self._enter_auto_state('SCANNING')
            return

        if self.auto_state == 'SCANNING':
            if not self._state_action_triggered:
                scan_end_pos = [SCAN_HALF_CIRCLE_TH1_END] + list(SCAN_POSITION[1:])
                self.start_arm_movement(self.controller.joint_pos, scan_end_pos, SCAN_ROTATE_DURATION)
                self._state_action_triggered = True
                self._best_capture = None
                self._pending_category = None
                self._pending_confirm_count = 0
                if self.detection_thread is not None:
                    self.detection_thread.set_active(True)  # 打开后台连续检测
                self.status_display.append("[自动采集] 开始关节限位内半圈旋转扫描检测(后台线程连续检测)...")
                return

            # 检测本身完全由后台线程(ForensicDetectionThread)连续进行，
            # 结果通过 on_detection_ready() 信号异步回传，这里只需要等旋转到位
            if not self.is_moving and not self.is_pausing:
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

            if self._best_capture is not None:
                det, xyz, normal, captured_depth_img, captured_th1 = self._best_capture
                self.dispatch_action_for_detection(det, xyz, normal, captured_depth_img, captured_th1)
            else:
                self.status_display.append("[自动采集] 本轮扫描未发现可执行的物证")
                self.statusBar().showMessage("本轮扫描未发现物证，重新开始")

            self._best_capture = None
            self._enter_auto_state('INIT_HOMING' if self.auto_mode else 'IDLE')
            return

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：execute_action — 将夹爪控制整合到运动序列中，添加等待时间参数
    # ──────────────────────────────────────────────────────────────────────────
    def execute_action(self, action_num, mask=None, depth_img=None, th1=None):
            """执行指定的动作(1,2或3)
            动作1(涂抹，液体类)：抓取点固定为棉签工具位，涂抹点为笛卡尔空间输入的坐标；
                RRT转移到目标点上方，若提供mask则按CDG-ATRG分割区域生成蛇形往复
                擦拭路径，否则退回固定偏移来回涂抹兜底；完成后RRT转移到
                EVIDENCE_RECOVERY_POINT释放棉签
            动作2(粘取，微小类)：抓取点固定为粘取器工具位，粘取点为笛卡尔空间输入的坐标；
                RRT转移到目标点上方，垂直下压-停留-提起完成粘附；完成后RRT转移到
                EVIDENCE_RECOVERY_POINT释放粘取器
            动作3(抓取，固体类)：抓取点即笛卡尔空间输入的目标点本身，沿表面法向量
                接近后真正闭合夹爪抓取（★不同于本文件早期版本/原文件里的同名动作，
                那里实际只在目标点打开夹爪、并未真正抓取，见下方"原代码"注释对照）；
                完成后RRT转移到EVIDENCE_RECOVERY_POINT松开夹爪放下物证

            mask/depth_img/th1: 仅液体类(action_num==1)用于生成蛇形擦拭路径，
                由自动检测流程(dispatch_action_for_detection)传入CDG-ATRG分割
                掩模及其对应的深度图/关节1角度；手动模式下(UI按钮直接点击)
                不传这三个参数，保持默认None，行为与之前完全一致。
            """
            if self.current_action is not None:
                QMessageBox.warning(self, "警告", "当前已有动作正在执行!")
                return

            try:
                safe_height = float(self.height_input.text())
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请输入有效的安全高度!")
                return

            # 从笛卡尔空间输入框读取坐标
            try:
                target_x = float(self.x_input.text())
                target_y = float(self.y_input.text())
                target_z = float(self.z_input.text())
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请在笛卡尔空间控制框中输入有效的 X / Y / Z 坐标!")
                return

            # ★ 读取法向量（用于涂抹/粘取/拍摄阶段）
            try:
                nx = float(self.nx_input.text())
                ny = float(self.ny_input.text())
                nz = float(self.nz_input.text())
                normal = [nx, ny, nz]
            except ValueError:
                normal = None

            q6_text = self.j6_input.text()
            q6 = float(q6_text) if q6_text.strip() else None

            # 根据动作类型设置抓取点
            if action_num == 1:  # 涂抹动作
                grab_x, grab_y, grab_z = -0.11, 0.175, 0.06
            elif action_num == 2:  # 粘取动作
                grab_x, grab_y, grab_z = 0.02, 0.18, 0.05
            elif action_num == 3:  # 抓取动作（固体类，抓取点即检测到的目标点本身）
                grab_x, grab_y, grab_z = target_x, target_y, target_z
            else:
                QMessageBox.warning(self, "错误", "无效的动作编号!")
                return

            # ============================================================
            # ★ 抓取相关：使用默认姿态（不传递法向量）
            # ============================================================
            # 对抓取点求逆运动学
            success_p2, joints_p2 = self.controller.move_to_cartesian(
                grab_x, grab_y, grab_z, q6
                # 不传递 normal → 使用默认姿态
            )
            if not success_p2:
                QMessageBox.warning(self, "错误",
                    f"无法对抓取点 ({grab_x}, {grab_y}, {grab_z}) 求解逆运动学!")
                return

            # 对抓取点安全高度求逆运动学
            success_above, joints_above_p2 = self.controller.move_to_cartesian(
                grab_x, grab_y, safe_height, q6
                # 不传递 normal → 使用默认姿态
            )
            if not success_above:
                QMessageBox.warning(self, "错误",
                    f"无法对抓取点安全高度 ({grab_x}, {grab_y}, {safe_height}) 求解逆运动学!")
                return

            # ============================================================
            # ★ 目标点（涂抹/粘取/拍摄）：传递法向量
            # ============================================================
            # 对目标点求逆运动学
            success_target, joints_target = self.controller.move_to_cartesian(
                target_x, target_y, target_z, q6, normal
                # 传递 normal → 使用法向量生成姿态
            )
            if not success_target:
                QMessageBox.warning(self, "错误",
                    f"无法对目标点 ({target_x}, {target_y}, {target_z}) 求解逆运动学!")
                return

            # ★★★ 新增：先把接触点原始坐标存下来，下面 target_x/y/z 马上要被沿法向量回退
            # 0.05m 改写成"进退点"坐标，move_linear() 需要用到两个坐标都不一样的点 ★★★
            contact_x, contact_y, contact_z = target_x, target_y, target_z

            target_x -= 0.05*normal[0]
            target_y -= 0.05*normal[1]
            target_z -= 0.05*normal[2]

            # 对目标点安全高度求逆运动学
            success_above_target, joints_above_target = self.controller.move_to_cartesian(
                target_x, target_y,target_z , q6, normal
                # 传递 normal → 使用法向量生成姿态
            )
            if not success_above_target:
                QMessageBox.warning(self, "错误",
                    f"无法对目标点安全高度 ({target_x}, {target_y}, {safe_height}) 求解逆运动学!")
                return

            # ============================================================
            # ★★★ 新增：物证回收位 —— 求解逆运动学（默认姿态，释放动作对
            # 末端朝向没有严格要求，不传法向量更简单也更安全）
            # ============================================================
            recovery_x, recovery_y, recovery_z = EVIDENCE_RECOVERY_POINT
            success_recovery, joints_recovery = self.controller.move_to_cartesian(
                recovery_x, recovery_y, recovery_z, q6
            )
            if not success_recovery:
                QMessageBox.warning(self, "错误",
                    f"无法对物证回收位 ({recovery_x}, {recovery_y}, {recovery_z}) 求解逆运动学!")
                return

            success_recovery_above, joints_recovery_above = self.controller.move_to_cartesian(
                recovery_x, recovery_y, safe_height, q6
            )
            if not success_recovery_above:
                QMessageBox.warning(self, "错误",
                    f"无法对物证回收位安全高度 ({recovery_x}, {recovery_y}, {safe_height}) 求解逆运动学!")
                return

            # 定义各点（关节解，仍保留：给"回零位"这类关节空间过渡用）
            point0 = [0, 0, 0, 0, 0, 0]
            point2       = joints_p2
            above_point2 = joints_above_p2
            point_target       = joints_target
            above_point_target = joints_above_target

            # ★★★ 新增：笛卡尔坐标点位，下面全部motion改用 move_linear() 走直线 ★★★
            p_grab         = [grab_x, grab_y, grab_z]
            p_grab_above   = [grab_x, grab_y, safe_height]
            p_target       = [contact_x, contact_y, contact_z]   # 涂抹/粘取/拍摄的接触点
            p_target_above = [target_x, target_y, target_z]      # 沿法向量回退0.05m后的进退点
            p_recovery       = [recovery_x, recovery_y, recovery_z]   # ★ 新增：物证回收释放点
            p_recovery_above = [recovery_x, recovery_y, safe_height]  # ★ 新增：回收位安全高度

            self.current_action = action_num
            self.enable_action_buttons(False)

            # ★★★ 新增：本函数改为同步阻塞执行——move_linear()/move_joint() 都会阻塞到运动结束
            # 才返回，所以不再需要 self.move_sequence 队列和 QTimer 异步调度。
            # 注意：阻塞期间界面（包括"紧急停止"按钮）不会响应，直到整个动作跑完。★★★
            publisher = self.controller.publisher
            ik = self.controller.move_to_cartesian

            try:
                # ════════════════════════════════════════════════════════
                # 原代码（保留对照，取工具这一段现改为"仅液体/微小类执行"，
                # 固体类抓取点就是目标点本身，不需要先去工具架取工具，
                # 改由下面 action_num==3 分支自己处理）：
                # # 1. 打开夹爪
                # self._send_gripper(420)
                # time.sleep(0.8)
                # # 3. 移动到抓取点上方 —— 起点是"当前关节角"，笛卡尔坐标未知，用 move_joint() 过渡
                # move_joint(publisher, self.controller.joint_pos.copy(), above_point2, MOVE_DURATION, controller=self.controller)
                # # 4. 下探到抓取点
                # move_linear(publisher, ik,  p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6)
                # # 5. 闭合夹爪
                # self._send_gripper(500)
                # time.sleep(1.0)
                # # 6. 返回安全高度
                # move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6)
                # ════════════════════════════════════════════════════════
                if action_num != 3:
                    # 液体/微小类：先去工具架取棉签/粘取器（默认姿态，不涉及法向约束）
                    # 1. 打开夹爪
                    self._send_gripper(420)
                    time.sleep(0.8)

                    # 3. 移动到抓取点上方 —— 起点是"当前关节角"，笛卡尔坐标未知，用 move_joint() 过渡
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point2, MOVE_DURATION, controller=self.controller)

                    # 4. 下探到抓取点 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 5. 闭合夹爪，取工具
                    self._send_gripper(500)
                    time.sleep(1.0)

                    # 6. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                if action_num == 1:  # 涂抹动作（液体类）
                    # ════════════════════════════════════════════════════
                    # 原代码（保留对照，已替换为下方"RRT转移段 + 按需蛇形
                    # 擦拭路径 + 转移到回收位释放"的新逻辑）：
                    # # 7. 移动到目标点（涂抹点）上方
                    # move_linear(publisher, ik, p_grab_above, p_target_above, MOVE_DURATION, q6=q6, normal=normal)
                    # # 8. 下探到目标点（涂抹点）
                    # move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)
                    # # 9. 在目标点来回涂抹
                    # example_smear_with_linear_motion(publisher, ik, p_target, q6=q6, normal=normal)
                    # # 10. 返回安全高度（涂抹结束后停在p_target，见上）
                    # move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)
                    # # 11. 返回抓取点上方
                    # move_linear(publisher, ik, p_target_above, p_grab_above, MOVE_DURATION, q6=q6)
                    # # 12. 下探到抓取点
                    # move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6)
                    # # 13. 打开夹爪（放回工具架）
                    # self._send_gripper(420)
                    # time.sleep(0.8)
                    # # 14. 返回安全高度
                    # move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6)
                    # # 15. 返回初始位置
                    # move_joint(publisher, above_point2, point0, MOVE_DURATION)
                    # self.controller.joint_pos = point0.copy()
                    # self.controller.q6_value = 0.0
                    # ════════════════════════════════════════════════════

                    # 7. 转移段：抓取点上方 -> 目标点(物证区域)上方 —— ★ 改用RRT规划
                    execute_cartesian_rrt_transfer(publisher, ik, p_grab_above, p_target_above,
                                                    MOVE_DURATION, q6=q6, normal=normal)

                    # 8. 接近段：法向约束下探到目标点，不经过RRT，保持不变
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 9. 在物证区域内擦拭 —— ★ 优先按CDG-ATRG分割掩模生成蛇形往复路径，
                    #    没有掩模(手动模式)或掩模太小/深度无效时，退回固定偏移来回涂抹兜底
                    snake_waypoints = None
                    if mask is not None and depth_img is not None:
                        snake_waypoints = pcalib.generate_snake_wipe_waypoints(
                            mask, depth_img, self.camera_intrinsics, th1=th1)

                    if snake_waypoints:
                        self.status_display.append(
                            "[涂抹] 按物证区域生成蛇形擦拭路径，共{}个路径点".format(len(snake_waypoints)))
                        current_point = p_target
                        for wp in snake_waypoints:
                            move_linear(publisher, ik, current_point, list(wp), 1.5, q6=q6, normal=normal)
                            current_point = list(wp)
                        # 擦拭完成后回到目标点中心，收尾位置与原逻辑保持一致
                        move_linear(publisher, ik, current_point, p_target, 1.5, q6=q6, normal=normal)
                    else:
                        self.status_display.append("[涂抹] 未获取到有效物证区域掩模，使用固定偏移来回涂抹兜底")
                        example_smear_with_linear_motion(publisher, ik, p_target, q6=q6, normal=normal)

                    # 10. 返回安全高度
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改用RRT规划，不再放回原工具架
                    execute_cartesian_rrt_transfer(publisher, ik, p_target_above, p_recovery_above,
                                                    MOVE_DURATION, q6=q6, normal=normal)

                    # 12. 下探到回收位 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 13. 打开夹爪，释放棉签(连同沾染的液体物证)到回收位
                    self._send_gripper(420)
                    time.sleep(0.8)

                    # 14. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 15. 返回初始位置
                    move_joint(publisher, joints_recovery_above, point0, MOVE_DURATION, controller=self.controller)
                    # ★★★ move_linear/move_joint是无状态的纯函数，不会像原来的
                    # update_arm_movement()那样自动更新self.controller.joint_pos，
                    # 这里手动同步一下，避免后续"移动到位置"等功能拿到过期的当前位置 ★★★
                    self.controller.joint_pos = point0.copy()
                    self.controller.q6_value = 0.0

                elif action_num == 2:  # 粘取动作（微小类）
                    # ════════════════════════════════════════════════════
                    # 原代码（保留对照，已替换为下方"RRT转移段 + 转移到
                    # 回收位释放"的新逻辑，粘附本身的下压-停留-提起不变）：
                    # # 7. 移动到目标点（粘取点）上方
                    # move_linear(publisher, ik, p_grab_above, p_target_above, MOVE_DURATION, q6=q6, normal=normal)
                    # # 8. 下探到目标点（粘取点）
                    # move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)
                    # # 9. 在目标点停留（模拟粘取动作）
                    # time.sleep(1.0)
                    # # 10. 返回安全高度
                    # move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)
                    # # 11. 返回抓取点上方
                    # move_linear(publisher, ik, p_target_above, p_grab_above, MOVE_DURATION, q6=q6)
                    # # 12. 下探到抓取点
                    # move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6)
                    # # 13. 打开夹爪（放回工具架）
                    # self._send_gripper(420)
                    # time.sleep(0.8)
                    # # 14. 返回安全高度
                    # move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6)
                    # # 15. 返回初始位置
                    # move_joint(publisher, above_point2, point0, MOVE_DURATION)
                    # self.controller.joint_pos = point0.copy()
                    # self.controller.q6_value = 0.0
                    # ════════════════════════════════════════════════════

                    # 7. 转移段：抓取点上方 -> 目标点上方 —— ★ 改用RRT规划
                    execute_cartesian_rrt_transfer(publisher, ik, p_grab_above, p_target_above,
                                                    MOVE_DURATION, q6=q6, normal=normal)

                    # 8. 接近段：法向约束下探到目标点，不经过RRT，保持不变
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 9. 在目标点停留（模拟粘取动作），不变
                    time.sleep(1.0)

                    # 10. 返回安全高度
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改用RRT规划，不再放回原工具架
                    execute_cartesian_rrt_transfer(publisher, ik, p_target_above, p_recovery_above,
                                                    MOVE_DURATION, q6=q6, normal=normal)

                    # 12. 下探到回收位 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 13. 打开夹爪，释放粘取器(连同粘附的微小物证)到回收位
                    self._send_gripper(420)
                    time.sleep(0.8)

                    # 14. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 15. 返回初始位置
                    move_joint(publisher, joints_recovery_above, point0, MOVE_DURATION, controller=self.controller)
                    self.controller.joint_pos = point0.copy()  # ★ 同步位置，理由同上
                    self.controller.q6_value = 0.0

                elif action_num == 3:  # 抓取动作（固体类）
                    # ════════════════════════════════════════════════════
                    # 原代码（保留对照，★已整体替换★）：原逻辑复用上面公共
                    # 前缀在p_grab(=目标点本身)以默认姿态闭合过一次夹爪，
                    # 这里又下探到同一个p_grab、且是"打开"夹爪(对应原注释
                    # "拍摄动作")，等于把刚抓到的东西又松开，并未真正带着
                    # 物证离开、也没有用到共同前缀里已经算好的、带法向量的
                    # p_target/p_target_above：
                    # # 7. 在抓取点上方停留（模拟拍摄）
                    # move_linear(publisher, ik, p_grab_above, p_grab_above, 5.0, q6=q6)
                    # # 8. 下探到抓取点
                    # move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6)
                    # # 9. 打开夹爪
                    # self._send_gripper(440)
                    # time.sleep(0.8)
                    # # 10. 返回安全高度
                    # move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6)
                    # # 11. 返回初始位置
                    # move_joint(publisher, above_point2, point0, MOVE_DURATION)
                    # self.controller.joint_pos = point0.copy()
                    # self.controller.q6_value = 0.0
                    # ════════════════════════════════════════════════════
                    # 新逻辑：固体类抓取点即目标点本身，没有"先去工具架"这一
                    # 段（见上面 if action_num != 3 的跳过），直接从当前位置
                    # (关节空间过渡，起点未知)移动到目标点上方的法向约束进退点
                    # p_target_above，再沿表面法向量插值接近到接触点p_target，
                    # 真正闭合夹爪抓取，抓稳后转移到回收位松开。

                    # 6a. 打开夹爪 —— 跳过了公共前缀的步骤1(取工具时才打开)，
                    #     这里补上，确保抓取前夹爪处于张开状态(不依赖上一次
                    #     动作结束时夹爪被留在什么角度)
                    self._send_gripper(420)
                    time.sleep(0.8)

                    # 7. 转移段：当前位置 -> 目标点上方(法向约束进退点) —— 起点笛卡尔坐标未知，仍用move_joint()过渡
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point_target, MOVE_DURATION, controller=self.controller)

                    # 8. 接近段：沿表面法向量下探到接触点 —— 法向约束，不经过RRT
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 9. 闭合夹爪，真正抓取物证（原代码这里是打开夹爪，并未真正抓取）
                    self._send_gripper(500)
                    time.sleep(1.0)

                    # 10. 沿法向量退回进退点
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 11. 转移段：目标点上方 -> 物证回收位上方 —— ★ 改用RRT规划
                    execute_cartesian_rrt_transfer(publisher, ik, p_target_above, p_recovery_above,
                                                    MOVE_DURATION, q6=q6, normal=normal)

                    # 12. 下探到回收位 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery_above, p_recovery, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 13. 打开夹爪，松开固体物证到回收位
                    self._send_gripper(420)
                    time.sleep(0.8)

                    # 14. 返回安全高度 —— ★ 补 normal 保持夹爪向下
                    move_linear(publisher, ik, p_recovery, p_recovery_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 15. 返回初始位置
                    move_joint(publisher, joints_recovery_above, point0, MOVE_DURATION, controller=self.controller)
                    self.controller.joint_pos = point0.copy()  # ★ 同步位置，理由同上
                    self.controller.q6_value = 0.0

            finally:
                # ★★★ 新增：无论运动是否顺利完成，都恢复按钮和状态显示（原async版本里
                # 这一步是在 update_arm_movement() 队列跑完之后才做，现在改成同步做）★★★
                self.current_action = None
                self.enable_action_buttons(True)
                self.show_current_status()

    def enable_action_buttons(self, enable):
        for btn in self.action_buttons:
            btn.setEnabled(enable)

    def start_arm_movement(self, start_pos, end_pos, duration):
        self.is_moving = True
        self.is_pausing = False
        self.move_start_time = rospy.get_time()
        self.move_duration = duration
        self.move_start_pos = start_pos
        self.move_end_pos = end_pos

    def rotate_j6(self, angle):
        new_pos = self.controller.joint_pos.copy()
        new_pos[5] += angle
        self.start_arm_movement(self.controller.joint_pos, new_pos, MOVE_DURATION / 2)
        self.controller.q6_value = new_pos[5]
        self.show_current_status()

    def return_to_zero(self):
        # 如果正在扫描，先停止扫描
        if self.is_scanning:
            self.is_scanning = False
        self.start_arm_movement(self.controller.joint_pos, INITIAL_POSITION, MOVE_DURATION)
        self.controller.q6_value = 0.0
        self.show_current_status()

    def start_arm(self):
        if self.launch_process.state() == QProcess.NotRunning:
            try:
                env = QProcessEnvironment.systemEnvironment()
                env.insert("PATH", os.environ["PATH"])
                env.insert("ROS_MASTER_URI", os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
                env.insert("ROS_PACKAGE_PATH", os.environ.get("ROS_PACKAGE_PATH", ""))
                self.launch_process.setProcessEnvironment(env)
                self.launch_process.start("roslaunch", ["gluon", "display.launch"])
                self.statusBar().showMessage("正在启动机械臂...")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"启动机械臂失败: {str(e)}")
        else:
            QMessageBox.warning(self, "提示", "机械臂已经启动!")

    def emergency_stop(self):
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
            self.statusBar().showMessage("机械臂已紧急停止!")
        else:
            QMessageBox.warning(self, "提示", "机械臂未启动!")

        # 停止扫描
        self.is_scanning = False

        # ★ 新增：紧急停止时同步终止自动采集流程
        self.auto_mode = False
        self.auto_state = 'IDLE'
        self._state_action_triggered = False
        self._best_capture = None
        if self.detection_thread is not None:
            self.detection_thread.set_active(False)

        self.is_moving = False
        self.is_pausing = False
        self.is_gripper_waiting = False  # ★ 新增：重置夹爪等待状态
        self.move_sequence = []
        self.current_move_index = 0
        self.current_action = None
        self.enable_action_buttons(True)
        self.show_current_status()

    def start_camera(self):
        if not self.camera_processes:
            try:
                # ★ 与原版 gluon_controller_kk_copy.py start_camera 完全一致:
                #   roslaunch forensic3_detector_v4 forensic3_detect.launch
                #   (含 astra 相机 + v4 检测节点; forensic3_detect.launch 已改
                #    enable_depth:=true, 同时提供 color/depth/camera_info/result_image)
                #   自动定位用 on_raw_depth 订阅 /camera/depth/image_raw 取深度,
                #   UI 相机窗用 result_image 显示带检测框画面——与原版行为一致。
                detect_cmd = (
                    "source /opt/ros/melodic/setup.bash && "
                    "source ~/forensic_ws/devel/setup.bash --extend && "
                    "source ~/forensic3_detect_ws_v4/devel/setup.bash --extend && "
                    "roslaunch forensic3_detector_v4 forensic3_detect.launch display:=false vis_thresh:=0.60"
                )
                detect_process = QProcess()
                detect_process.setProcessChannelMode(QProcess.MergedChannels)
                detect_process.start("bash", ["-c", detect_cmd])
                self.camera_processes.append(detect_process)

                # UI 相机窗订阅 v4 检测结果图 (与原版一致, 让相机窗显示画面)
                self.controller.image_subscriber = rospy.Subscriber(
                    "/forensic3_detect_v4/result_image",
                    Image,
                    self.controller.handle_camera_image
                )
                self.statusBar().showMessage("物证检测: 相机+v4检测已启动 (vis_thresh=0.60)")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"启动相机失败: {str(e)}")
        else:
            QMessageBox.warning(self, "提示", "相机已经启动!")

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

            self.camera_label.setText("相机已停止")
            self.statusBar().showMessage("相机系统已停止!")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：closeEvent — 先释放资源再弹确认框
    # ──────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        # 停止扫描
        self.is_scanning = False

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

        # ★ 统一通过 port_handler 关闭端口
        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            self.port_handler = None
            self.servo_controller = None

        # 停止机械臂
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
            if not self.launch_process.waitForFinished(1000):
                self.launch_process.kill()

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
class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller')
        self.publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.joint_pos = INITIAL_POSITION.copy()
        self.q6_value = 0.0
        self.camera_image = None
        self.image_subscriber = None

    def handle_camera_image(self, msg):
        self.camera_image = msg

    def move_to_cartesian(self, x, y, z, q6=None, normal=None):
        """
        移动到笛卡尔空间位置，可指定表面法向量
        
        参数:
            x, y, z: 位置坐标
            q6: 关节6角度（可选）
            normal: 表面法向量 [nx, ny, nz]（可选）
        
        返回:
            success: 是否成功
            target_joints: 目标关节角
        """
        target_pos = [x, y, z]
        
        # 处理姿态：如果有法向量则使用法向量转换，否则使用默认姿态
        if normal is not None:
            try:
                nx, ny, nz = normal
                target_orn = normal_to_quaternion(nx, ny, nz)
                print(f"[调试] 使用法向量 ({nx}, {ny}, {nz}) 生成姿态: {target_orn}")
            except Exception as e:
                rospy.logerr(f"法向量转换失败: {e}")
                return False, None
        else:
            # 默认姿态
            target_orn = [0, 0.7071, 0, 0.7071]
            print(f"[调试] 使用默认姿态: {target_orn}")

        print(f"\n[调试] 目标位置: {target_pos}")
        print(f"[调试] 目标姿态: {target_orn}")

        # 检查 workSpace
        try:
            ws_result = workSpace(target_pos)
            print(f"[调试] workSpace 结果: {ws_result}")
        except Exception as e:
            print(f"[调试] workSpace 调用失败: {e}")
            ws_result = workSpace(target_pos)
        
        if not ws_result:
            rospy.logerr("目标位置超出工作空间")
            return False, None

        # 调用逆运动学
        print(f"[调试] 调用 inv_k...")
        solutions = inv_k(target_pos, target_orn)
        print(f"[调试] 求解结果: {len(solutions)} 组解")
        
        if not solutions:
            rospy.logerr("逆运动学求解失败")
            return False, None
        
        # ★ 取与当前关节最接近的解 (与 forensic_ws 一致, 抑制 th1 翻转/关节突跳)
        target_joints = inv_k_best(target_pos, target_orn, ref_joints=self.joint_pos)
        if target_joints is None:
            target_joints = solutions[0]
        print(f"[调试] 使用最优解(连续): {target_joints}")

        if q6 is not None:
            self.q6_value = q6
        target_joints[5] = self.q6_value

        # ★ 关键: 同步更新 joint_pos, 使 move_linear 下一帧 inv_k_best 的
        #   ref_joints 是当前帧实际关节角(而非过期值), 抑制解分支跳变/机械臂扭曲。
        #   move_linear/move_joint 是无状态纯函数不会自动更新 joint_pos, 这里补上。
        self.joint_pos = list(target_joints)

        return True, target_joints

    def rotate_joint6(self, angle):
        self.q6_value += angle
        new_pos = self.joint_pos.copy()
        new_pos[5] = self.q6_value
        return True

    def return_to_zero(self):
        rospy.loginfo("正在返回初始位置...")
        self.joint_pos = INITIAL_POSITION.copy()
        self.q6_value = 0.0
        rospy.loginfo("已回到初始位置")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def pub2ros_pos(publisher, pos):
    msg = JointState()
    msg.name = JOINT_NAMES
    msg.position = pos
    publisher.publish(msg)


def interpolate(start, end, fraction):
    return [s + (e - s) * fraction for s, e in zip(start, end)]


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：直线点到点运动函数（自包含、无状态、不依赖UI/move_sequence，方便整段搬到别的脚本里用）
# ──────────────────────────────────────────────────────────────────────────────
def move_linear(publisher, ik_solver, start_xyz, end_xyz, duration=5.0, rate_hz=30,
                 q6=None, normal=None):
    """
    笛卡尔空间"直线点到点"运动：让末端从 start_xyz 沿直线运动到 end_xyz。

    只依赖 rospy 和一个逆运动学求解回调，不依赖任何UI状态、move_sequence队列，
    整个函数(连同下面用到的 pub2ros_pos)可以直接复制到别的脚本里用。

    ★ 这是一个阻塞函数：调用后会原地阻塞约 duration 秒，期间按 rate_hz 频率
      不断插值+发布，直到运动结束才返回。如果在PyQt按钮点击的回调里直接调用，
      这段时间界面会卡住无响应；需要不卡界面的话，把这次调用放到QThread里跑，
      或者用简单的多线程/多进程去调，函数本身不用改。

    参数:
        publisher : rospy.Publisher，发布 sensor_msgs/JointState 到 /joint_states
        ik_solver : 逆运动学求解函数，签名须为 (x, y, z, q6, normal) -> (success: bool, joints: list[6])
                    直接传 ArmController.move_to_cartesian 这个绑定方法即可
        start_xyz : [x, y, z] 直线起点，单位米
        end_xyz   : [x, y, z] 直线终点，单位米
        duration  : 运动总时长，单位秒
        rate_hz   : 插值/发布频率，单位Hz
        q6        : 全程保持不变的关节6角度(弧度)；为None则用ik_solver内部当前值
        normal    : 全程保持不变的末端姿态法向量[nx,ny,nz]；为None则用ik_solver默认姿态

    返回:
        True 表示插补循环正常跑完；期间个别帧逆解失败只会跳过并打印警告，不会中断整体运动
    """
    steps = max(1, int(duration * rate_hz))
    rate = rospy.Rate(rate_hz)

    for i in range(steps + 1):
        fraction = i / float(steps)
        x = round(start_xyz[0] + (end_xyz[0] - start_xyz[0]) * fraction, 4)
        y = round(start_xyz[1] + (end_xyz[1] - start_xyz[1]) * fraction, 4)
        z = round(start_xyz[2] + (end_xyz[2] - start_xyz[2]) * fraction, 4)

        success, joints = ik_solver(x, y, z, q6, normal)
        if success:
            pub2ros_pos(publisher, joints)
        else:
            rospy.logwarn(f"move_linear: 插补点({x:.3f},{y:.3f},{z:.3f})逆解失败，跳过该帧")

        rate.sleep()

    return True


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：关节空间"点到点"运动函数 —— 和 move_linear() 同一风格的阻塞函数，
# 用在终点没有已知笛卡尔坐标的场合（比如回到关节零位），两者配套使用
# ──────────────────────────────────────────────────────────────────────────────
def move_joint(publisher, start_joints, end_joints, duration=5.0, rate_hz=30,
               controller=None):
    """
    关节空间"点到点"运动：直接对6个关节角线性插值再发布，不经过逆运动学。
    末端走的路径不是直线，只用于终点没有已知笛卡尔坐标的过渡（例如回零位）。
    和 move_linear() 一样是阻塞函数，调用后原地阻塞到运动结束才返回。
    controller: 可选, 传入 ArmController 实例则每帧同步 joint_pos (供后续 inv_k_best 连续择优)。
    """
    steps = max(1, int(duration * rate_hz))
    rate = rospy.Rate(rate_hz)

    for i in range(steps + 1):
        fraction = i / float(steps)
        current = interpolate(start_joints, end_joints, fraction)
        pub2ros_pos(publisher, current)
        # ★ 同步 joint_pos, 使后续 move_linear 的 inv_k_best ref_joints 是当前实际关节角
        if controller is not None:
            controller.joint_pos = list(current)
            controller.q6_value = current[5]
        rate.sleep()

    return True


# ──────────────────────────────────────────────────────────────────────────────
# ★★★ 新增：转移段专用 —— 用 rrt_planner.plan_path() 做笛卡尔空间安全段
# 路径规划，取代execute_action()里原来"起点直接一条move_linear到终点"的
# 做法。RRT只用于"转移段"（起止两端都是已知笛卡尔坐标、末端姿态在转移
# 过程中不需要严格保持某个方向的自由空间移动），不用于"接近段"（法向约束
# 接近物证表面，姿态必须全程对准法向量，仍然用 move_linear(..., normal=)
# 直接插值，不经过RRT）。
# ──────────────────────────────────────────────────────────────────────────────
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
    try:
        controller = ArmController()
        app = QApplication(sys.argv)
        ui = ArmControllerUI(controller)
        ui.show()
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
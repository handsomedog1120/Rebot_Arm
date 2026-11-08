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
from sensor_msgs.msg import JointState, Image
from functions import workSpace, inv_k, inv_k_best
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox, QTextEdit,
                             QMessageBox, QSplitter, QFormLayout, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtGui import QColor
import time

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

        # 现在调用 show_current_status 时所有变量都已定义
        self.show_current_status()

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

        # 三个预设点坐标 (可编辑, 对应动作1/2/3 的核心点位)
        preset_layout = QFormLayout()
        # 涂抹预设点 = 动作1的涂抹接触点 (p_target), 同时控制 p_grab_above 的 X/Y
        self.preset1_x = QLineEdit("-0.185"); self.preset1_y = QLineEdit("0.035"); self.preset1_z = QLineEdit("0.08")
        preset1_row = QHBoxLayout(); preset1_row.addWidget(QLabel("涂抹预设点:")); preset1_row.addWidget(QLabel("X")); preset1_row.addWidget(self.preset1_x)
        preset1_row.addWidget(QLabel("Y")); preset1_row.addWidget(self.preset1_y); preset1_row.addWidget(QLabel("Z")); preset1_row.addWidget(self.preset1_z)
        preset_layout.addRow(preset1_row)
        # 粘取预设点 = 动作2的抓取点 (p_grab), 同时控制 p_grab_above 的 X/Y
        self.preset2_x = QLineEdit("-0.097"); self.preset2_y = QLineEdit("0.05"); self.preset2_z = QLineEdit("0.05")
        preset2_row = QHBoxLayout(); preset2_row.addWidget(QLabel("粘取预设点:")); preset2_row.addWidget(QLabel("X")); preset2_row.addWidget(self.preset2_x)
        preset2_row.addWidget(QLabel("Y")); preset2_row.addWidget(self.preset2_y); preset2_row.addWidget(QLabel("Z")); preset2_row.addWidget(self.preset2_z)
        preset_layout.addRow(preset2_row)
        # 抓取预设点 = 动作3的最终放置点 (p3_target), 同时控制 p3_top 的 X/Y
        self.preset3_x = QLineEdit("-0.03"); self.preset3_y = QLineEdit("0.165"); self.preset3_z = QLineEdit("0.05")
        preset3_row = QHBoxLayout(); preset3_row.addWidget(QLabel("抓取预设点:")); preset3_row.addWidget(QLabel("X")); preset3_row.addWidget(self.preset3_x)
        preset3_row.addWidget(QLabel("Y")); preset3_row.addWidget(self.preset3_y); preset3_row.addWidget(QLabel("Z")); preset3_row.addWidget(self.preset3_z)
        preset_layout.addRow(preset3_row)
        action_layout.addLayout(preset_layout)

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
        ws_layout.addWidget(QLabel("X轴范围: -0.12 ~ 0.30 (米)"))
        ws_layout.addWidget(QLabel("Y轴范围: -0.10 ~ 0.30 (米)"))
        ws_layout.addWidget(QLabel("Z轴范围: 0.05 ~ 0.25 (米)"))
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

        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMinimumHeight(200)

        status_layout.addWidget(self.status_display)
        status_group.setLayout(status_layout)

        layout.addWidget(camera_group)
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

                # 相机窗直接显示订阅到的图像 (v3 检测节点已在图像上画了检测框+类别+置信度)
                if channels == 3 and msg.encoding == 'rgb8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                elif channels == 3 and msg.encoding == 'bgr8':
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
        
        status_text = f"""当前机械臂状态:

    当前执行动作: {action_text}
    关节角度 (弧度):
    关节1: {self.controller.joint_pos[0]:.4f}
    关节2: {self.controller.joint_pos[1]:.4f}
    关节3: {self.controller.joint_pos[2]:.4f}
    关节4: {self.controller.joint_pos[3]:.4f}
    关节5: {self.controller.joint_pos[4]:.4f}
    关节6: {self.controller.q6_value:.4f}

    工作空间状态:
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
    # ★ 修复版：execute_action — 将夹爪控制整合到运动序列中，添加等待时间参数
    # ──────────────────────────────────────────────────────────────────────────
    def execute_action(self, action_num):
            """执行指定的动作(1,2或3)
            动作1(涂抹)：抓取点固定为 (-0.02, 0.15, 0.2)，涂抹点为笛卡尔空间输入的坐标
            动作2(粘取)：抓取点固定为 (0.02, 0.15, 0.2)，粘取点为笛卡尔空间输入的坐标
            动作3(拍摄)：抓取点为笛卡尔空间输入的坐标
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

            # ★ 读取三个预设点 (从 UI 输入框)
            p1x = float(self.preset1_x.text()); p1y = float(self.preset1_y.text()); p1z = float(self.preset1_z.text())  # 涂抹预设点
            p2x = float(self.preset2_x.text()); p2y = float(self.preset2_y.text()); p2z = float(self.preset2_z.text())  # 粘取预设点
            p3x = float(self.preset3_x.text()); p3y = float(self.preset3_y.text()); p3z = float(self.preset3_z.text())  # 抓取预设点

            # 根据动作类型设置抓取点
            if action_num == 1:  # 涂抹动作 (套用粘取逻辑, 抓取点 = 涂抹预设点)
                grab_x, grab_y, grab_z = p1x, p1y, p1z
                a1_p3x, a1_p3y, a1_p3z = 0.0, 0.0, 0.0
            elif action_num == 2:  # 粘取动作
                grab_x, grab_y, grab_z = p2x, p2y, p2z
                a1_p3x, a1_p3y, a1_p3z = 0.0, 0.0, 0.0
            elif action_num == 3:  # 抓取动作
                # 抓取点 = 笛卡尔控制区的 target; 最终放置点 (p3) = 抓取预设点
                grab_x, grab_y, grab_z = target_x, target_y, target_z
                a1_p3x, a1_p3y, a1_p3z = p3x, p3y, p3z
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
            
            success_p3,above_point3 = self.controller.move_to_cartesian(
                a1_p3x, a1_p3y, safe_height, q6
                # 不传递 normal → 使用默认姿态
            )
            if not success_p3:
                QMessageBox.warning(self, "错误",
                    f"无法对动作1目标点 ({a1_p3x}, {a1_p3y}, {safe_height}) 求解逆运动学!")
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
            

            # 定义各点（关节解，仍保留：给"回零位"这类关节空间过渡用）
            point0 = [0, 0, 0, 0, 0, 0]
            point2       = joints_p2
            above_point2 = joints_above_p2
            point_target       = joints_target
            above_point_target = joints_above_target

            # ★★★ 笛卡尔坐标点位, 预设点 X/Y 联动"上方点"的 X/Y (Z 保留安全高度) ★★★
            p_grab         = [grab_x, grab_y, grab_z]
            p_grab_above   = [grab_x, grab_y, safe_height]
            # 动作1/2: p_target = 笛卡尔控制区的接触点 (与粘取一致); 动作3: 同理
            p_target       = [contact_x, contact_y, contact_z]
            p_target_above = [target_x, target_y, target_z]
            # 动作3: p3_target = 抓取预设点; p3上方 X/Y 跟随
            if action_num == 3:
                p3_target_above = [p3x, p3y, safe_height]
                p3_target = [p3x, p3y, p3z]
            else:
                p3_target_above = [a1_p3x, a1_p3y, safe_height]
                p3_target = [a1_p3x, a1_p3y, a1_p3z]

            self.current_action = action_num
            self.enable_action_buttons(False)

            # ★★★ 新增：本函数改为同步阻塞执行——move_linear()/move_joint() 都会阻塞到运动结束
            # 才返回，所以不再需要 self.move_sequence 队列和 QTimer 异步调度。
            # 注意：阻塞期间界面（包括"紧急停止"按钮）不会响应，直到整个动作跑完。★★★
            publisher = self.controller.publisher
            ik = self.controller.move_to_cartesian

            try:
                if action_num != 3:

                    # 1. 打开夹爪
                    self._send_gripper(350)
                    time.sleep(0.8)

                    # 3. 移动到抓取点上方 —— 起点是"当前关节角"，笛卡尔坐标未知，用 move_joint() 过渡
                    move_joint(publisher, self.controller.joint_pos.copy(), above_point2, MOVE_DURATION)


                    # 4. 下探到抓取点 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik,  p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 5. 闭合夹爪
                    self._send_gripper(500)
                    time.sleep(1.0)

                    # 6. 返回安全高度 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    if action_num == 1:  # 涂抹动作 (套用粘取, 仅在第9步加往复运动)
                        # 7. 移动到目标点（涂抹点）上方
                        move_joint(publisher, above_point2, joints_above_target, MOVE_DURATION)

                        # 8. 下探到目标点（涂抹点）
                        move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 9. 在目标点往复涂抹 (小幅X方向来回)
                        smear_cx = p_target[0]; smear_cy = p_target[1]; smear_cz = p_target[2]
                        smear_amp = 0.015
                        smear_pts = [
                            [smear_cx + smear_amp, smear_cy, smear_cz],
                            [smear_cx - smear_amp, smear_cy, smear_cz],
                            [smear_cx + smear_amp, smear_cy, smear_cz],
                            [smear_cx - smear_amp, smear_cy, smear_cz],
                            [smear_cx + smear_amp, smear_cy, smear_cz],
                            [smear_cx, smear_cy, smear_cz],
                        ]
                        sp = p_target
                        for ep in smear_pts:
                            move_linear(publisher, ik, sp, ep, 2.0, q6=q6, normal=normal)
                            sp = ep

                        # 10. 返回安全高度
                        move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 11. 返回抓取点上方
                        move_joint(publisher, joints_above_target, above_point2, MOVE_DURATION)

                        # 12. 下探到抓取点
                        move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 13. 打开夹爪
                        self._send_gripper(350)
                        time.sleep(0.8)

                        # 14. 返回安全高度
                        move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 15. 返回初始位置
                        move_joint(publisher, above_point2, point0, MOVE_DURATION)
                        self.controller.joint_pos = point0.copy()
                        self.controller.q6_value = 0.0

                    elif action_num == 2:  # 粘取动作
                        # 7. 移动到目标点（粘取点）上方 —— ★ 替换为 move_linear() 直线
                        # move_linear(publisher, ik, p_grab_above, p_target_above, MOVE_DURATION, q6=q6, normal=normal)
                        move_joint(publisher, above_point2, joints_above_target, MOVE_DURATION)

                        # 8. 下探到目标点（粘取点） —— ★ 替换为 move_linear() 直线
                        move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 9. 在目标点停留（模拟粘取动作）
                        time.sleep(1.0)

                        # 10. 返回安全高度 —— ★ 替换为 move_linear() 直线
                        move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 11. 返回抓取点上方 —— ★ 替换为 move_linear() 直线
                        # move_linear(publisher, ik, p_target_above, p_grab_above, MOVE_DURATION, q6=q6, normal=normal)
                        move_joint(publisher, joints_above_target, above_point2, MOVE_DURATION)

                        # 12. 下探到抓取点 —— ★ 替换为 move_linear() 直线
                        move_linear(publisher, ik, p_grab_above, p_grab, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 13. 打开夹爪
                        self._send_gripper(350)
                        time.sleep(0.8)

                        # 14. 返回安全高度 —— ★ 替换为 move_linear() 直线
                        move_linear(publisher, ik, p_grab, p_grab_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                        # 15. 返回初始位置 —— 终点是关节零位，仍用 move_joint() 过渡
                        move_joint(publisher, above_point2, point0, MOVE_DURATION)
                        self.controller.joint_pos = point0.copy()  # ★ 新增：同步位置，理由同上
                        self.controller.q6_value = 0.0

                elif action_num == 3:  # 拍摄动作
                     # 1. 打开夹爪
                    self._send_gripper(350)
                    time.sleep(0.8)

                    # 3. 移动到抓取点上方 —— 起点是"当前关节角"，笛卡尔坐标未知，用 move_joint() 过渡
                    move_joint(publisher,self.controller.joint_pos.copy() , joints_above_target, MOVE_DURATION)


                    # 4. 下探到抓取点 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik, p_target_above, p_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 5. 闭合夹爪
                    self._send_gripper(500)
                    time.sleep(1.0)

                    # 6. 返回安全高度 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik, p_target, p_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)
                    
                    # 7. 在抓取点上方停留（模拟拍摄） —— ★ 替换为 move_linear()（起止点相同，原地停留）
                    time.sleep(3)     
                    

                    move_joint(publisher, joints_above_target, above_point3, MOVE_DURATION)
                    
                    # 8. 下探到抓取点 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik, p3_target_above, p3_target, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 9. 打开夹爪
                    self._send_gripper(300)
                    time.sleep(0.8)

                    # 10. 返回安全高度 —— ★ 替换为 move_linear() 直线
                    move_linear(publisher, ik, p3_target, p3_target_above, MOVE_DURATION / 2, q6=q6, normal=normal)

                    # 11. 返回初始位置 —— 终点是关节零位，仍用 move_joint() 过渡
                    move_joint(publisher, above_point3, point0, MOVE_DURATION)
                    self.controller.joint_pos = point0.copy()  # ★ 新增：同步位置，理由同上
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
                    # ★ 物证检测模式: roslaunch forensic3_detector_v4 (含相机+v4检测节点, 带条形卷积+金字塔)
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

                    # UI 相机窗订阅 v4 检测结果图
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
                if process.state() == QProcess.Running:
                    process.terminate()
                    if not process.waitForFinished(1000):
                        process.kill()
                process.close()
            self.camera_processes = []

            if hasattr(self.controller, 'image_subscriber'):
                self.controller.image_subscriber.unregister()
                delattr(self.controller, 'image_subscriber')

            self.camera_label.setText("相机已停止")
            self.statusBar().showMessage("物证检测相机已停止!")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 修复版：closeEvent — 先释放资源再弹确认框
    # ──────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        # 停止扫描
        self.is_scanning = False
        
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
            from gluon.ros_gluon.gluon.scripts.functions import workSpace as ws2
            ws_result = ws2(target_pos)
            print(f"[调试] 使用 functions.workSpace 结果: {ws_result}")
        
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
        
        # ★ 取与当前关节最接近的解 (避免 th1 翻转/关节突跳)
        target_joints = inv_k_best(target_pos, target_orn, ref_joints=self.joint_pos)
        if target_joints is None:
            target_joints = solutions[0]
        print(f"[调试] 使用最优解(连续): {target_joints}")

        if q6 is not None:
            self.q6_value = q6
        target_joints[5] = self.q6_value

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
def move_joint(publisher, start_joints, end_joints, duration=5.0, rate_hz=30):
    """
    关节空间"点到点"运动：直接对6个关节角线性插值再发布，不经过逆运动学。
    末端走的路径不是直线，只用于终点没有已知笛卡尔坐标的过渡（例如回零位）。
    和 move_linear() 一样是阻塞函数，调用后原地阻塞到运动结束才返回。
    """
    steps = max(1, int(duration * rate_hz))
    rate = rospy.Rate(rate_hz)

    for i in range(steps + 1):
        fraction = i / float(steps)
        current = interpolate(start_joints, end_joints, fraction)
        pub2ros_pos(publisher, current)
        rate.sleep()

    return True


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
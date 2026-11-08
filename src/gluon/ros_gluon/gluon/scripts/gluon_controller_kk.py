#!/usr/bin/env python3
import serial.tools.list_ports  
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import rospy
import numpy as np
from scservo_sdk import scscl as BaseSCSCL
from scservo_sdk import *
from PyQt5.QtCore import (QProcessEnvironment, Qt, QProcess, QTimer, QThread, pyqtSignal)
env = QProcessEnvironment()
from sensor_msgs.msg import JointState, Image
from functions import workSpace, inv_k
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox, QTextEdit, 
                             QMessageBox, QSplitter,  QFormLayout,QComboBox )
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
MOVE_DURATION = 5.0  # 单次移动时间(秒)
PAUSE_DURATION = 2.0 # 暂停时间(秒)
INITIAL_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始位置
SAFE_HEIGHT = 0.2    # 安全高度H(米)

class EnhancedSCSCL(BaseSCSCL):
    """扩展舵机控制类，添加速度控制方法"""
    
    def WriteSpeed(self, scs_id, speed, acc=0):
        """
        SCS速度控制协议扩展
        :param speed: 速度值(单位: 0.074°/s)
        :param acc: 加速度(0-255)
        """
        # 构造数据包（基于原库协议格式）
        txpacket = [
            0xFF, 0xFF, scs_id, 0x07, 0x03, 0x20,
            speed & 0xFF, (speed >> 8) & 0xFF,
            acc & 0xFF, (acc >> 8) & 0xFF
        ]
        checksum = (~sum(txpacket[2:])) & 0xFF
        txpacket.append(checksum)
        
        # 调用基类发送方法
        self.portHandler.write(txpacket)
        return self.read_response()

class ServoScanThread(QThread):
    found_servo = pyqtSignal(int, str)

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name
        self.running = True

    def run(self):
        try:
            portHandler = PortHandler(self.port_name)
            if portHandler.openPort() and portHandler.setBaudRate(115200):
                packetHandler = scscl(portHandler)
                for scs_id in range(1, 254):  # 舵机ID范围通常为1-253
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
        self.current_action = None  # 初始化current_action属性
        self.action_buttons = []    # 确保action_buttons已初始化
        self.servo_controller = None  # 新增关键属性初始化
        self.scan_thread = None     # 舵机扫描线程
        self.setWindowTitle("Gluon 机械臂控制系统")
        self.setGeometry(100, 100, 1920, 1080)  # 增大窗口尺寸
        
        # 科技风配色方案
        self.COLOR_BG = QColor(20, 30, 50)        # 深蓝背景
        self.COLOR_PANEL = QColor(40, 50, 70)     # 面板颜色
        self.COLOR_ACCENT = QColor(0, 180, 255)   # 强调色
        self.COLOR_TEXT = QColor(255, 255, 255)   # 白色文字

        # 设置全局样式
        self.setStyleSheet(f"""
            /* 全局基础样式 */
            QMainWindow, QWidget {{
                color: {self.COLOR_TEXT.name()};
                background-color: {self.COLOR_BG.name()};
            }}

            /* 分组框及标题 */
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

            /* 所有文本控件 */
            QLabel, QLineEdit, QTextEdit, QPushButton {{
                color: {self.COLOR_TEXT.name()};
            }}

            /* 输入框特别设置 */
            QLineEdit {{
                border: 1px solid {self.COLOR_ACCENT.name()};
                border-radius: 4px;
                padding: 8px;
                background-color: {self.COLOR_PANEL.name()};
            }}

            /* 按钮交互状态 */
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

            /* 状态显示区域 */
            QTextEdit {{
                background-color: {self.COLOR_PANEL.name()};
                border: 1px solid {self.COLOR_ACCENT.name()};
                border-radius: 5px;
                padding: 10px;
            }}
        """)

        # 主布局优化
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 强化标题
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
        
        # 相机相关进程
        self.camera_processes = []
        self.init_ui()
        self.show_current_status()
        
        # 使用单个定时器处理所有定时任务
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_all)
        self.timer.start(30)  # 约30fps
        
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
        self.init_servo_ui()

    def init_servo_ui(self):
        # 创建舵机控制主分组框
        servo_group = QGroupBox("SC-2500舵机控制")
        servo_layout = QVBoxLayout()
        port_group = QGroupBox("通信配置")
        port_layout = QFormLayout()
        
        # 端口选择
        self.port_combo = QComboBox()
        self.btn_refresh = QPushButton("刷新端口")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        
        # 舵机扫描
        self.btn_scan = QPushButton("扫描舵机")
        self.btn_scan.clicked.connect(self.start_servo_scan)
        self.servo_list = QComboBox()
        
        port_layout.addRow("通信端口:", self.port_combo)
        port_layout.addRow(self.btn_refresh)
        port_layout.addRow("扫描舵机:", self.btn_scan)
        port_layout.addRow("检测到舵机:", self.servo_list)
        port_group.setLayout(port_layout)

        # ----------------------------
        # 2. 舵机参数控制模块
        # ----------------------------
        control_group = QGroupBox("舵机控制")
        control_layout = QFormLayout()
        
        # 角度控制
        self.angle_input = QLineEdit("300")  # 夹爪打开
        self.btn_set_angle = QPushButton("设置角度")
        self.btn_set_angle.clicked.connect(self.set_servo_angle)
        
        control_layout.addRow("目标角度:", self.angle_input)
        control_layout.addRow(self.btn_set_angle)
        control_group.setLayout(control_layout)

        # 在角度控制下方添加夹爪控制按钮
        gripper_group = QGroupBox("夹爪控制")
        gripper_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("打开夹爪 (300)")
        self.btn_open.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_open.clicked.connect(lambda: self.set_gripper_angle(300) )
        
        self.btn_close = QPushButton("闭合夹爪 (500)")
        self.btn_close.setStyleSheet("background-color: #F44336; color: white;")
        self.btn_close.clicked.connect(lambda: self.set_gripper_angle(500))
        
        gripper_layout.addWidget(self.btn_open)
        gripper_layout.addWidget(self.btn_close)
        gripper_group.setLayout(gripper_layout)
        
        control_layout.addRow(gripper_group)  # 添加到原有控制布局

        # ----------------------------
        # 组合布局
        # ----------------------------       
        servo_layout.addWidget(port_group)
        servo_layout.addWidget(control_group)
        servo_group.setLayout(servo_layout)
 

        # 添加到主界面左侧布局
        left_panel = self.centralWidget().findChild(QWidget)
        if left_panel and left_panel.layout():
            left_panel.layout().insertWidget(2, servo_group)
        else:
            self.centralWidget().layout().addWidget(servo_group)

    # 新增夹爪角度设置方法
    def set_gripper_angle(self, angle):
        self.angle_input.setText(str(angle))
        self.set_servo_angle()

    def set_servo_angle(self):
        try:
            angle = int(self.angle_input.text())
            if not 0 <= angle <= 4096:
                raise ValueError("角度值需在0-4096之间")
                
            # 获取当前选择的舵机ID
            if self.servo_list.currentIndex() < 0:
                raise ValueError("请先扫描并选择舵机")
                
            scs_id = self.servo_list.currentData()
            
            # 确保扫描线程已停止
            if self.scan_thread and self.scan_thread.isRunning():
                self.scan_thread.stop()
                time.sleep(0.1)  # 等待端口释放
            
            # 初始化舵机控制器
            self.init_servo_connection()
            
            if self.servo_controller:
                # 发送指令并添加延时
                scs_comm_result, scs_error = self.servo_controller.WritePos(
                    scs_id, 
                    angle, 
                    0,     # 时间参数（单位：ms）
                    int(100 / 0.074)  # 示例：将100 RPM转换为协议速度值
                )
                time.sleep(0.05)  # 等待舵机响应
                
                if scs_comm_result == COMM_SUCCESS:
                    QMessageBox.information(self, "成功", "角度设置成功!")
                else:
                    QMessageBox.warning(self, "错误", f"通信失败: {self.servo_controller.getTxRxResult(scs_comm_result)}")
            else:
                raise ConnectionError("舵机控制器未初始化")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def init_servo_connection(self):
        """初始化舵机通信连接"""
        if not self.port_combo.currentText():
            raise ConnectionError("请先选择通信端口")
            
        # 如果已经连接则跳过
        if self.servo_controller and self.servo_controller.portHandler.is_open:
            return
            
        # 创建新的端口处理器
        portHandler = PortHandler(self.port_combo.currentText())
        
        # 尝试打开端口，最多重试3次
        for attempt in range(3):
            if portHandler.openPort() and portHandler.setBaudRate(115200):
                self.servo_controller = scscl(portHandler)
                return
            else:
                # 关闭可能部分打开的端口
                if portHandler.is_open:
                    portHandler.closePort()
                time.sleep(0.1)  # 等待端口释放
        
        # 如果三次尝试都失败
        raise ConnectionError("无法打开舵机端口，请检查端口是否被占用")

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        
    def start_servo_scan(self):
        self.servo_list.clear()
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "警告", "请先选择通信端口!")
            return
            
        # 停止之前的扫描线程
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread.wait(500)  # 等待线程结束
            
        # 关闭舵机控制器端口（如果打开）
        if self.servo_controller and self.servo_controller.portHandler.is_open:
            self.servo_controller.portHandler.closePort()
            
        self.scan_thread = ServoScanThread(port)
        self.scan_thread.found_servo.connect(self.update_servo_list)
        self.scan_thread.start()
        
    def update_servo_list(self, scs_id, info):
        self.servo_list.addItem(info, scs_id)
        
    def update_all(self):
        """处理所有定时更新任务"""
        self.update_camera_view()
        self.update_arm_movement()
    
    def update_arm_movement(self):
        """非阻塞式的机械臂移动更新"""
        if not self.is_moving and not self.is_pausing:
            if self.current_move_index < len(self.move_sequence):
                # 开始下一个移动
                move = self.move_sequence[self.current_move_index]
                self.start_arm_movement(move['start'], move['end'], move['duration'])
                self.current_move_index += 1
                return
        
        if self.is_moving:
            current_time = rospy.get_time()
            elapsed = current_time - self.move_start_time
            fraction = min(elapsed / self.move_duration, 1.0)
            
            # 计算当前位置
            current_pos = interpolate(self.move_start_pos, self.move_end_pos, fraction)
            pub2ros_pos(self.controller.publisher, current_pos)
            
            # 更新控制器中的当前位置
            self.controller.joint_pos = current_pos
            self.controller.q6_value = current_pos[5]
            
            # 检查是否完成移动
            if fraction >= 1.0:
                self.is_moving = False
                self.is_pausing = True
                self.pause_start_time = current_time
                self.show_current_status()
        
        elif self.is_pausing:
            current_time = rospy.get_time()
            if current_time - self.pause_start_time >= PAUSE_DURATION:
                self.is_pausing = False
                # 检查是否完成整个动作序列
                if self.current_move_index >= len(self.move_sequence):
                    self.move_sequence = []
                    self.current_move_index = 0
                    self.current_action = None
                    self.enable_action_buttons(True)
                    self.show_current_status()
    
    def init_ui(self):
        # 主窗口布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 使用分割器创建左右两部分
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        
        # 右侧相机和状态面板
        right_panel = self.create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)  # 左侧占2份
        splitter.setStretchFactor(1, 3)  # 右侧占3份
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
        # 添加底部状态栏
        self.statusBar().showMessage("系统就绪")
    
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)
        
        # 1. 机械臂控制面板
        control_group = QGroupBox("机械臂控制")
        control_layout = QVBoxLayout()
        
        # 1.1 笛卡尔空间控制
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
        
        move_btn = QPushButton("移动到位置")
        move_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        move_btn.clicked.connect(self.move_to_position)
        cartesian_layout.addRow(move_btn)
        
        cartesian_group.setLayout(cartesian_layout)
        
        # 1.2 动作序列控制
        action_group = QGroupBox("动作序列控制")
        action_layout = QVBoxLayout()
        
        # 安全高度设置
        height_layout = QHBoxLayout()
        height_label = QLabel("安全高度H (米):")
        self.height_input = QLineEdit(str(SAFE_HEIGHT))
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)
        action_layout.addLayout(height_layout)
        
        # 动作按钮
        btn_layout1 = QHBoxLayout()
        self.action1_btn = QPushButton("执行动作1")
        self.action1_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action1_btn.clicked.connect(lambda: self.execute_action(1))
        btn_layout1.addWidget(self.action1_btn)
        
        btn_layout2 = QHBoxLayout()
        self.action2_btn = QPushButton("执行动作2")
        self.action2_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action2_btn.clicked.connect(lambda: self.execute_action(2))
        btn_layout2.addWidget(self.action2_btn)
        
        btn_layout3 = QHBoxLayout()
        self.action3_btn = QPushButton("执行动作3")
        self.action3_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.action3_btn.clicked.connect(lambda: self.execute_action(3))
        btn_layout3.addWidget(self.action3_btn)
        
        action_layout.addLayout(btn_layout1)
        action_layout.addLayout(btn_layout2)
        action_layout.addLayout(btn_layout3)
        
        # 添加到动作按钮列表
        self.action_buttons = [self.action1_btn, self.action2_btn, self.action3_btn]
        
        action_group.setLayout(action_layout)
        
        # 1.3 关节空间控制
        joint_group = QGroupBox("关节空间控制")
        joint_layout = QVBoxLayout()
        
        # 关节6旋转控制
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
        
        # 预设位置按钮
        preset_layout = QHBoxLayout()
        home_btn = QPushButton("返回初始位置")
        home_btn.setStyleSheet("background-color: #FF9800; color: white;")
        home_btn.clicked.connect(self.return_to_zero)
        
        preset_layout.addWidget(home_btn)
        joint_layout.addLayout(preset_layout)
        
        joint_group.setLayout(joint_layout)
        
        # 1.4 系统控制
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
        
        # 添加到控制面板
        control_layout.addWidget(cartesian_group)
        control_layout.addWidget(action_group)
        control_layout.addWidget(joint_group)
        control_layout.addWidget(system_group)
        control_group.setLayout(control_layout)
        
        # 2. 工作空间信息
        ws_group = QGroupBox("工作空间限制")
        ws_layout = QVBoxLayout()
        ws_layout.addWidget(QLabel("X轴范围: 0.15 ~ 0.40 (米)"))
        ws_layout.addWidget(QLabel("Y轴范围: -0.10 ~ 0.30 (米)"))
        ws_layout.addWidget(QLabel("Z轴范围: 0.10 ~ 0.35 (米)"))
        ws_group.setLayout(ws_layout)
        
        # 添加到左侧面板
        layout.addWidget(control_group)
        layout.addWidget(ws_group)
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """创建右侧相机和状态面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. 相机视图
        camera_group = QGroupBox("相机视图")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setText("等待相机图像...")
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        
        # 相机控制按钮
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
        
        # 2. 状态信息
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()
        
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMinimumHeight(200)
        
        status_layout.addWidget(self.status_display)
        status_group.setLayout(status_layout)
        
        # 添加到右侧面板
        layout.addWidget(camera_group)
        layout.addWidget(status_group)
        
        return panel
    
    def update_camera_view(self):
        """完全绕过cv_bridge的图像处理方法"""
        if hasattr(self.controller, 'camera_image') and self.controller.camera_image is not None:
            try:
                # 直接解析ROS Image消息
                msg = self.controller.camera_image
                
                # 检查图像编码
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
                
                # 将数据转换为numpy数组
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(
                    msg.height, msg.width, channels)
                
                # 转换为RGB格式
                if channels == 3 and msg.encoding == 'bgr8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                elif channels == 1:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                
                # 转换为Qt图像
                h, w, ch = cv_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    cv_image.data, 
                    w, h, 
                    bytes_per_line, 
                    QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.width(),  # 使用实际宽度
                    self.camera_label.height(),
                    Qt.KeepAspectRatioByExpanding,  # 保持比例扩展
                    Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)
                
            except Exception as e:
                self.camera_label.setText(f"图像处理错误: {str(e)}")
                rospy.logerr(f"更新相机视图错误: {str(e)}")
    
    def show_current_status(self):
        action_text = "无" if self.current_action is None else f"动作{self.current_action}"
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
"""
        self.status_display.setPlainText(status_text)
        self.statusBar().showMessage(f"系统运行中 - 当前动作: {action_text}")
    
    def move_to_position(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            
            q6_text = self.j6_input.text()
            q6 = float(q6_text) if q6_text else None
            
            success, joints = self.controller.move_to_cartesian(x, y, z, q6)
            
            if success:
                # 使用非阻塞方式移动机械臂
                self.start_arm_movement(self.controller.joint_pos, joints, MOVE_DURATION)
                QMessageBox.information(self, "成功", "机械臂开始移动!")
            else:
                QMessageBox.warning(self, "错误", "无法移动到目标位置!")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字!")
    
    def execute_action(self, action_num):
        """执行指定的动作(1,2或3)"""
        if self.current_action is not None:
            QMessageBox.warning(self, "警告", "当前已有动作正在执行!")
            return
        
        try:
            safe_height = float(self.height_input.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的安全高度!")
            return
        
        # 定义四个目标点位
        point0 = [0, 0, 0, 0, 0, 0] # 起点
        point1 = [0.0, -0.8, 0.8, 0.0, 1.4, 0.0] 
        point2 = [0.5, -0.8, 0.8, 0.0, 1.4, 0.0] 
        point3 = [1.0, -0.8, 0.8, 0.0, 1.4, 0.0]  
        point4 = [-1.0, -0.8, 0.8, 0.0, 1.4, 0.0] 
        point5 = [0.8, -0.8, 0.8, 0.0, 1.4, 0.0] 
        
        # 清空当前移动序列
        self.move_sequence = []
        self.current_move_index = 0
        self.current_action = action_num
        self.enable_action_buttons(False)
        
        # 获取当前位置
        current_pos = self.controller.joint_pos.copy()
        
        # 1. 设置夹爪为打开状态(300)
        self.set_gripper_angle(300)
        
        # 2. 上升到安全高度(如果当前位置低于安全高度)
        if current_pos[2] < safe_height:
            above_current = current_pos.copy()
            above_current[2] = safe_height
            self.move_sequence.append({
                'start': current_pos,
                'end': above_current,
                'duration': MOVE_DURATION/2
            })
            current_pos = above_current.copy()
        
        # 3. 移动到目标点2上方
        above_point2 = point2.copy()
        above_point2[2] = safe_height
        self.move_sequence.append({
            'start': current_pos,
            'end': above_point2,
            'duration': MOVE_DURATION
        })
        
        # 4. 下探到目标点2
        self.move_sequence.append({
            'start': above_point2,
            'end': point2,
            'duration': MOVE_DURATION/2
        })
        
        # 5. 设置夹爪为闭合状态(500)
        self.set_gripper_angle(500)
        
        # 6. 返回到安全高度
        self.move_sequence.append({
            'start': point2,
            'end': above_point2,
            'duration': MOVE_DURATION/2
        })
        
        # 根据不同的动作执行不同的序列
        if action_num == 1:
            # 动作1：移动到点3，来回移动五次
            above_point3 = point3.copy()
            above_point3[2] = safe_height
            
            # 7. 移动到目标点3上方
            self.move_sequence.append({
                'start': above_point2,
                'end': above_point3,
                'duration': MOVE_DURATION
            })
            
            # 8. 下探到目标点3
            self.move_sequence.append({
                'start': above_point3,
                'end': point3,
                'duration': MOVE_DURATION/2
            })
            
            # 9. 在点3和点5之间来回移动五次
            for i in range(5):
                # 从点3到点4
                self.move_sequence.append({
                    'start': point3,
                    'end': point5,
                    'duration': MOVE_DURATION/2
                })
                # 从点4返回点3
                self.move_sequence.append({
                    'start': point5,
                    'end': point3,
                    'duration': MOVE_DURATION/2
                })
            
            # 10. 上移到安全高度
            self.move_sequence.append({
                'start': point3,
                'end': above_point3,
                'duration': MOVE_DURATION/2
            })
            
            # 11. 移动到目标点2上方
            self.move_sequence.append({
                'start': above_point3,
                'end': above_point2,
                'duration': MOVE_DURATION
            })
            
            # 12. 下探到目标点2
            self.move_sequence.append({
                'start': above_point2,
                'end': point2,
                'duration': MOVE_DURATION/2
            })
            
            # 13. 设置夹爪为打开状态(300)
            self.set_gripper_angle(300)
            
            # 14. 返回到安全高度
            self.move_sequence.append({
                'start': point2,
                'end': above_point2,
                'duration': MOVE_DURATION/2
            })
            
            # 15. 返回起点
            self.move_sequence.append({
                'start': above_point2,
                'end': point0,
                'duration': MOVE_DURATION
            })
            
        elif action_num == 2:
            # 动作2：移动到点4，停留1秒
            above_point4 = point4.copy()
            above_point4[2] = safe_height
            
            # 7. 移动到目标点4上方
            self.move_sequence.append({
                'start': above_point2,
                'end': above_point4,
                'duration': MOVE_DURATION
            })
            
            # 8. 下探到目标点4
            self.move_sequence.append({
                'start': above_point4,
                'end': point4,
                'duration': MOVE_DURATION/2
            })
            
            # 9. 在点4停留1秒
            self.move_sequence.append({
                'start': point4,
                'end': point4,
                'duration': 1.0  # 停留1秒
            })
            
            # 10. 上移到安全高度
            self.move_sequence.append({
                'start': point4,
                'end': above_point4,
                'duration': MOVE_DURATION/2
            })
            
            # 11. 移动到目标点2上方
            self.move_sequence.append({
                'start': above_point4,
                'end': above_point2,
                'duration': MOVE_DURATION
            })
            
            # 12. 下探到目标点2
            self.move_sequence.append({
                'start': above_point2,
                'end': point2,
                'duration': MOVE_DURATION/2
            })
            
            # 13. 设置夹爪为打开状态(300)
            self.set_gripper_angle(300)
            
            # 14. 返回到安全高度
            self.move_sequence.append({
                'start': point2,
                'end': above_point2,
                'duration': MOVE_DURATION/2
            })
            
            # 15. 返回起点
            self.move_sequence.append({
                'start': above_point2,
                'end': point0,
                'duration': MOVE_DURATION
            })
            
        elif action_num == 3:
            # 动作3：在安全高度停留5秒，然后返回点2
            # 6. 在安全高度停留5秒
            self.move_sequence.append({
                'start': above_point2,
                'end': above_point2,
                'duration': 5.0  # 停留5秒
            })
            
            # 7. 下探到目标点2
            self.move_sequence.append({
                'start': above_point2,
                'end': point2,
                'duration': MOVE_DURATION/2
            })
            
            # 8. 设置夹爪为打开状态(300)
            self.set_gripper_angle(300)
            
            # 9. 返回到安全高度
            self.move_sequence.append({
                'start': point2,
                'end': above_point2,
                'duration': MOVE_DURATION/2
            })
            
            # 10. 返回起点
            self.move_sequence.append({
                'start': above_point2,
                'end': point0,
                'duration': MOVE_DURATION
            })
        
        QMessageBox.information(self, "动作开始", f"开始执行动作{action_num}!")
    
    def enable_action_buttons(self, enable):
        """启用或禁用动作按钮"""
        for btn in self.action_buttons:
            btn.setEnabled(enable)
    
    def start_arm_movement(self, start_pos, end_pos, duration):
        """开始非阻塞式的机械臂移动"""
        self.is_moving = True
        self.is_pausing = False
        self.move_start_time = rospy.get_time()
        self.move_duration = duration
        self.move_start_pos = start_pos
        self.move_end_pos = end_pos
    
    def rotate_j6(self, angle):
        """单独旋转关节6"""
        new_pos = self.controller.joint_pos.copy()
        new_pos[5] += angle
        self.start_arm_movement(self.controller.joint_pos, new_pos, MOVE_DURATION/2)
        self.controller.q6_value = new_pos[5]
        self.show_current_status()
    
    def return_to_zero(self):
        """返回初始位置"""
        self.start_arm_movement(self.controller.joint_pos, INITIAL_POSITION, MOVE_DURATION)
        self.controller.q6_value = 0.0
        self.show_current_status()
    
    def start_arm(self):
        """启动机械臂"""
        if self.launch_process.state() == QProcess.NotRunning:
            try:
                # 确保使用正确的ROS环境
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
        """紧急停止机械臂"""
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
            self.statusBar().showMessage("机械臂已紧急停止!")
        else:
            QMessageBox.warning(self, "提示", "机械臂未启动!")
        
        # 停止任何正在进行的移动
        self.is_moving = False
        self.is_pausing = False
        self.move_sequence = []
        self.current_move_index = 0
        self.current_action = None
        self.enable_action_buttons(True)
        self.show_current_status()
    
    def start_camera(self):
       """启动相机系统 - 修改为不依赖cv_bridge"""
       if not self.camera_processes:
            try:
                # 启动astra相机
                astra_cmd = "roslaunch orbbec_camera astra_stereo_u3.launch"
                astra_process = QProcess()
                astra_process.start("bash", ["-c", f"source ~/ros_ws/devel/setup.bash && {astra_cmd}"])
                self.camera_processes.append(astra_process)
                
                # 启动aruco识别
                aruco_cmd = "roslaunch aruco_ros single.launch"
                aruco_process = QProcess()
                aruco_process.start("bash", ["-c", f"source ~/catkin_ws/devel/setup.bash && {aruco_cmd}"])
                self.camera_processes.append(aruco_process)
                
                # 订阅相机图像话题
                self.controller.image_subscriber = rospy.Subscriber(
                    "/aruco_single/result", 
                    Image, 
                    self.controller.handle_camera_image
                )
                
                self.statusBar().showMessage("相机系统已启动!")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"启动相机失败: {str(e)}")
       else:
            QMessageBox.warning(self, "提示", "相机已经启动!")
    
    def stop_camera(self):
        """停止相机系统"""
        if self.camera_processes:
            for process in self.camera_processes:
                if process.state() == QProcess.Running:
                    process.terminate()  # 先尝试正常终止
                    if not process.waitForFinished(1000):  # 等待1秒
                        process.kill()  # 如果进程还在运行，强制终止
                process.close()  # 释放资源
            self.camera_processes = []
            
            # 取消订阅
            if hasattr(self.controller, 'image_subscriber'):
                self.controller.image_subscriber.unregister()
                delattr(self.controller, 'image_subscriber')
            
            self.camera_label.setText("相机已停止")
            self.statusBar().showMessage("相机系统已停止!")
        else:
            QMessageBox.warning(self, "提示", "相机未启动!")
        
    def closeEvent(self, event):
        """重写关闭事件，确保所有进程都被终止"""
        # 停止扫描线程
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread.wait(500)  # 等待线程结束
            
        # 停止舵机控制器
        if hasattr(self, 'servo_controller') and self.servo_controller:
            if self.servo_controller.portHandler.is_open:
                self.servo_controller.portHandler.closePort()
            
        # 停止机械臂
        if self.launch_process.state() == QProcess.Running:
            self.launch_process.terminate()
            if not self.launch_process.waitForFinished(1000):
                self.launch_process.kill()
                
        # 停止相机
        self.stop_camera()
        
        # 关闭ROS节点
        rospy.signal_shutdown('UI closed')
        
        # 确认关闭
        reply = QMessageBox.question(self, '确认退出', 
                                   "确定要退出系统吗?", 
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)

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

class ArmController:
        def __init__(self):
            rospy.init_node('arm_controller')
            self.publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
            self.joint_pos = INITIAL_POSITION.copy()
            self.q6_value = 0.0
            self.camera_image = None
            self.image_subscriber = None
        
        def handle_camera_image(self, msg):
            """简化的图像处理回调"""
            self.camera_image = msg
        
        def move_to_cartesian(self, x, y, z, q6=None):
            target_pos = [x, y, z]
            
            if not workSpace(target_pos):
                rospy.logerr("目标位置超出工作空间")
                return False, None
            
            solutions = inv_k(target_pos)

            if not solutions:
                rospy.logerr("逆运动学求解失败")
                return False, None
            
            # 从逆解中选择一个解
            target_joints = solutions[0]
            
            # 如果提供了q6值，使用它；否则保持当前q6值
            if q6 is not None:
                self.q6_value = q6
            target_joints[5] = self.q6_value  # 设置关节6的值
            
            return True, target_joints
        
        def rotate_joint6(self, angle):
            """单独旋转关节6"""
            self.q6_value += angle
            new_pos = self.joint_pos.copy()
            new_pos[5] = self.q6_value
            return True
        
        def return_to_zero(self):
            """返回初始位置"""
            rospy.loginfo("正在返回初始位置...")
            self.joint_pos = INITIAL_POSITION.copy()
            self.q6_value = 0.0
            rospy.loginfo("已回到初始位置")
            return True

def pub2ros_pos(publisher, pos):
        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = pos
        publisher.publish(msg)

def interpolate(start, end, fraction):
        """线性插值"""
        return [s + (e - s) * fraction for s, e in zip(start, end)]
    
if __name__ == '__main__':
    try:
        controller = ArmController()
        app = QApplication(sys.argv)
        ui = ArmControllerUI(controller)
        ui.show()
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
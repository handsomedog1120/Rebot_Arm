#!/usr/bin/env python
# -*- coding: utf-8 -*-
from paho.mqtt import client as mqtt_client
import random
import threading
import rospy
from sensor_msgs.msg import JointState
from time import sleep
import math

# MQTT配置
broker = '192.168.0.108'
port = 1883
topic = '/joint_states'
client_id = 'python-mqtt-sub' + str(random.randint(0, 1000))

# 定义多个目标点位（关节角度）
waypoints = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 初始位置
    [0.5, 0.2, -0.3, 0.4, 0.1, 0.2],  # 点位1
    [-0.3, 0.4, 0.1, -0.2, 0.3, 0.1],  # 点位2
    [0.2, -0.1, 0.4, 0.3, -0.2, 0.3],  # 点位3
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 返回初始位置
]

# 运动参数
move_duration = 3.0  # 每个点位间移动时间(秒)
pause_duration = 1.0  # 到达点位后暂停时间(秒)

def pub2ros_pos(publisher, pos):
    msg = JointState()
    msg.name = ['axis_joint_1', 'axis_joint_2', 'axis_joint_3', 'axis_joint_4', 'axis_joint_5', 'axis_joint_6']
    msg.position = pos
    publisher.publish(msg)

def interpolate(start, end, fraction):
    """线性插值函数"""
    return [s + (e - s) * fraction for s, e in zip(start, end)]

def move_arm(publisher, start_pos, end_pos, duration):
    """平滑移动机械臂从start_pos到end_pos"""
    steps = int(duration * 10)  # 10Hz控制频率
    for i in range(steps + 1):
        fraction = i / float(steps)
        current_pos = interpolate(start_pos, end_pos, fraction)
        pub2ros_pos(publisher, current_pos)
        rospy.sleep(0.1)  # 控制频率

def ros_publisher():
    rospy.init_node('pos_publisher', anonymous=True)
    publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(1)  # 等待 publisher 建立连接

    current_pos = waypoints[0]  # 初始位置

    try:
        while not rospy.is_shutdown():
            for target_pos in waypoints:
                # 移动到目标点位
                move_arm(publisher, current_pos, target_pos, move_duration)
                current_pos = target_pos
                # 在目标点位暂停
                rospy.sleep(pause_duration)

            # 反向移动
            for target_pos in reversed(waypoints[:-1]):
                move_arm(publisher, current_pos, target_pos, move_duration)
                current_pos = target_pos
                rospy.sleep(pause_duration)

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    try:
        print("Starting waypoint control...")
        ros_publisher()
    except Exception as e:
        print(f"Error: {e}")
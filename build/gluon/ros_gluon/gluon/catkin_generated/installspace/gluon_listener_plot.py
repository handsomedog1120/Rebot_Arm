#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from paho.mqtt import client as mqtt_client
import random
import threading
# 发布/joint_states话题，消息类型sensor_msgs::JointState

import rospy
from sensor_msgs.msg import JointState
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
import time
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import threading

sensor_pos = []
q1 = []
time_ = []
sig = 0
shutdown = 0
start_time = 0

def thread_job():
    rospy.spin()

def ROScallback_fuc(msg):
    '''ROS订阅的回调函数'''
    global sensor_pos
    global q1
    global time_
    global start_time
    global sig
    global shutdown

    sensor_pos = msg.position
    if len(sensor_pos) == 6 and shutdown != 1:
        sig = 1

    if start_time != 0 and shutdown != 1:
        q1.append(sensor_pos[0])
        time_.append(time.time()-start_time)
        # print(sensor_pos)

def ros_subscriber():
    '''设置ROS订阅节点'''
    rospy.init_node('sensor_listener', anonymous=True)
    rospy.Subscriber("/sensor_jointstates", JointState, ROScallback_fuc)
    # rospy.spin()

if __name__ == '__main__':
    ros_subscriber()
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()

    while not rospy.is_shutdown():
        while sig == 1:
            global start_time
            global sig
            global shutdown

            start_time = time.time()
            sleep(5)
            q1 = np.array(q1)
            time_ = np.array(time_)
            plt.xlabel('time')
            plt.ylabel('q1')
            plt.title("t-q1")
            plt.plot(time_, q1)
            plt.show()

            sig = 0
            shutdown = 1

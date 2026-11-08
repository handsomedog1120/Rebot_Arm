#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from paho.mqtt import client as mqtt_client
import random
import threading
# 发布/joint_states话题，消息类型sensor_msgs::JointState

import rospy
from sensor_msgs.msg import JointState
from time import sleep
from functions import *

# hw
# broker = '192.168.8.193'

# jb16
#broker = '192.168.43.181'
#同一局域网下的ipv4 ifconfig查询
broker = '192.168.0.108'
port = 1883
topic = 'leapmotioninfo'
client_id = 'python-mqtt-sub' + str(random.randint(0, 1000))

def connect_mqtt():
    '''connect to mqtt server'''

    def on_connect(client, userdata, flags, rc):
        '''连接回调函数'''
        # 响应状态码为0表示连接成功
        if rc == 0:
            print("Connected to MQTT OK!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def pub2mqtt_pos(client, topic, pos):
    '''发送关节角给mqtt'''
    payload = 'gluonsensordata#' + str(pos[0]) + '#' + str(pos[1]) + '#' + str(pos[2]) + '#' + str(pos[3]) + '#' + str(
        pos[4]) + '#' + str(pos[5])
    client.publish(topic, payload)
    # print(payload)

def ROScallback_fuc(msg):
    '''ROS订阅的回调函数'''
    global sensor_pos
    global client

    sensor_pos = msg.position
    print(sensor_pos)
    pub2mqtt_pos(client, 'gluonsensordata', sensor_pos)

def ros_subscriber():
    '''设置ROS订阅节点'''
    rospy.init_node('sensor_listener', anonymous=True)
    rospy.Subscriber("/sensor_jointstates", JointState, ROScallback_fuc)
    rospy.spin()

if __name__ == '__main__':
    client = connect_mqtt()
    ros_subscriber()

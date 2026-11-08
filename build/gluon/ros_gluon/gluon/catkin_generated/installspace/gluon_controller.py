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

#jb16
#broker = '192.168.43.181'
#同一局域网下的ipv4 ifconfig查询
broker = '192.168.0.108'
port = 1883
topic = 'leapmotioninfo'
client_id = 'python-mqtt-sub' + str(random.randint(0, 1000))

# 控制关节6旋转
q6_control = 0
joint_pos = [0, 0, 0, 0, 0, 0]

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

def sub_send2ros(client, topic, publisher):
    '''订阅主题并接收消息'''
    def on_message(client, userdata, msg):
        payload = msg.payload.decode()

        # '''接受6joints控制信息'''
        # if 'model2robot' in payload:
        #     payload = payload.split('#')
        #     a1 = float(payload[1])
        #     a2 = float(payload[2])
        #     a3 = float(payload[3])
        #     a4 = float(payload[4])
        #     a5 = float(payload[5])
        #     a6 = float(payload[6])
        #     joint_pos = [a1, a2, a3, a4, a5, a6]
        #     pub2ros_pos(publisher, joint_pos)

        '''
        接受xyz控制信息
        @ leapmotionxyz#x#y#z#q6_incre
        '''
        if 'leapmotionxyz' in payload:
            payload = payload.split('#')
            hand_x = float(payload[1])
            hand_y = float(payload[2])
            hand_z = float(payload[3])
            target_pos = [hand_x, hand_y, hand_z]

            global q6_control
            global joint_pos

            if workSpace(target_pos):
                joint_pos = inv_k(target_pos)
                if len(joint_pos) != 0:
                    joint_pos1 = joint_pos[0]

                    joint_pos[5] = q6_control
                    print(joint_pos)
                    pub2ros_


                    pos(publisher, joint_pos)
                    pub2mqtt_pos(client, 'gluonsensorinfo', joint_pos)
                print(joint_pos)
        if 'leapmotionendeffector' in payload:
            payload = payload.split('#')
            q6_incre = float(payload[1])
            # gripper_pos = float(payload[2])

            # global q6_control
            # global joint_pos

            if -2.7 < q6_control + q6_incre < 2.7:
                q6_control += q6_incre
                joint_pos[5] = q6_control
                print(joint_pos)
                pub2ros_pos(publisher, joint_pos)
                pub2mqtt_pos(client, 'gluonsensorinfo', joint_pos)
            else:
                pass

        if 'leapmotioninit' in payload:
            payload = payload.split('#')
            a1 = float(payload[1])
            a2 = float(payload[2])
            a3 = float(payload[3])
            a4 = float(payload[4])
            a5 = float(payload[5])
            a6 = float(payload[6])

            joint_pos = [a1, a2, a3, a4, a5, a6]
            print(joint_pos)
            pub2ros_pos(publisher, joint_pos)
            pub2mqtt_pos(client, 'gluonsensorinfo', joint_pos)


    # 订阅指定消息主题
    client.subscribe(topic)
    client.on_message = on_message

def pub2mqtt_pos(client, topic, pos):
    '''发送关节角给mqtt'''
    payload = 'gluonsensorinfo#'+str(pos[0])+'#'+str(pos[1])+'#'+str(pos[2])+'#'+str(pos[3])+'#'+str(pos[4])+'#'+str(pos[5])
    client.publish(topic, payload)

def pub2ros_pos(publisher_name, pos):
    # 初始化sensor_msgs::JointStates类型的消息
    msg = JointState()
    msg.name = ['axis_joint_1', 'axis_joint_2', 'axis_joint_3', 'axis_joint_4', 'axis_joint_5', 'axis_joint_6']
    msg.position = pos

    # 发布消息
    publisher_name.publish(msg)

def ros_publisher():
    # ROS节点初始化
    rospy.init_node('pos_publisher', anonymous=True)
    # 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
    publisher = rospy.Publisher('/joint_states', JointState, queue_size=1)

    # # 设置循环的频率Hz
    # rate = rospy.Rate(1)
    # 执行内容(发送的第一个消息不执行)
    pub2ros_pos(publisher, [0, 0, 0, 0, 0, 0])
    pub2ros_pos(publisher, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return publisher

if __name__ == '__main__':
    try:
        print("Start control")
        publisher = ros_publisher()
        client = connect_mqtt()
        sub_send2ros(client, topic, publisher)
        client.loop_forever()
        # t1 = threading.Thread(target=client.loop_start, args=())
        # t1.start()

    except rospy.ROSInterruptException:
        print('over')
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from paho.mqtt import client as mqtt_client
import random
import threading
# 发布/joint_states话题，消息类型sensor_msgs::JointState

import rospy
from sensor_msgs.msg import JointState
from time import sleep

def pub2ros_pos(publisher_name, pos):
    '''发送关节角给ros_gluon_node'''
    # 初始化sensor_msgs::JointStates类型的消息
    msg = JointState()
    msg.name = ['axis_joint_1', 'axis_joint_2', 'axis_joint_3', 'axis_joint_4', 'axis_joint_5', 'axis_joint_6']
    msg.position = pos

    # 发布消息
    publisher_name.publish(msg)

def ros_publisher():
    '''ros节点初始化'''
    # ROS节点初始化
    rospy.init_node('pos_publisher', anonymous=True)
    # 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
    publisher = rospy.Publisher('/joint_states', JointState, queue_size=1)

    # # 设置循环的频率Hz
    # rate = rospy.Rate(1)

    # 执行内容(发送的第一个消息不执行)
    pub2ros_pos(publisher, [0, 0, 0, 0, 0, 0])
    return publisher

def sleep_():
    sleep(0.2)

def main():
    pos1 = [1.22771799,0.294756,-1.276146,-0.148428,-1.454148,-0.385577999]
    # pos2 = [1.0850578528632322, 0.48477585780004873, -2.0112688943732815, -2.4960447521733302, -1.0850578528632322, 0.0]
    # pos3 = [0.99322932782642825, 0.49697586139216743, -1.9895696690568971, -2.4865455304490647, -0.99322932782642837, 0.0]
    # pos4 = [0.90842409924702894, 0.51693861708581723, -1.9538579785283985, -2.4707965956142157, -0.90842409924702894, 0.0]
    # pos5 = [0.83126444325038928, 0.54420345901126843, -1.9046989514985582, -2.4489024105098265, -0.83126444325038917, 0.0]
    # pos6 = [0.7618511036568083, 0.57828215322928422, -1.8426857961535867, -2.4209679493828711, -0.76185110365680841, 0.0]
    # pos7 = [0.69989400407580671, 0.61875908240295407, -1.7682979121868501, -2.3870569945898041, -0.69989400407580671, 0.0]
    # pos8 = [0.64485555061116551, 0.66537778305339579, -1.6817608831750497, -2.3471386662284455, -0.6448555506111654, 0.0]
    # pos9 = [0.59607338322928904, 0.71811939606865627, -1.5828992243713711, -2.3010186204400274, -0.59607338322928904, 0.0]
    # pos10 = [0.55284967905982207, 0.77729429812287265, -1.4709474754160234, -2.2482417735388962, -0.55284967905982207, 0.0]
    # pos11 = [0.51450788193194397, 0.84368916093925139, -1.3442433798780327, -2.1879325408172843, -0.51450788193194397, 0.0]

    publisher = ros_publisher()

    while not rospy.is_shutdown():
        pub2ros_pos(publisher, [0, 0, 0, 0, 0, 0])
        # sleep(1)
        pub2ros_pos(publisher, [0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
        sleep_()

        pub2ros_pos(publisher, pos1)
        sleep(1)
        # pub2ros_pos(publisher, pos2)
        # sleep_()
        # pub2ros_pos(publisher, pos3)
        # sleep_()
        # pub2ros_pos(publisher, pos4)
        # sleep_()
        # pub2ros_pos(publisher, pos5)
        # sleep_()
        # pub2ros_pos(publisher, pos6)
        # sleep_()
        # pub2ros_pos(publisher, pos7)
        # sleep_()
        # pub2ros_pos(publisher, pos8)
        # sleep_()
        # pub2ros_pos(publisher, pos9)
        # sleep_()
        # pub2ros_pos(publisher, pos10)
        # sleep_()
        # pub2ros_pos(publisher, pos11)
        # sleep_()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print('over')
        pass

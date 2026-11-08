#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState


def main():
    rospy.init_node('rebot_sim')
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    joint_state = JointState()
    joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
                        'joint_left', 'joint_right']
    joint_state.position = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    while not rospy.is_shutdown():
        joint_state.header.stamp = rospy.Time.now()
        pub.publish(joint_state)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


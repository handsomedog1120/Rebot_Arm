#!/usr/bin/env python
import rospy

def timer_callback(event):
    rospy.signal_shutdown("达到最大发布次数")

if __name__ == '__main__':
    rospy.init_node('shutdown_timer')
    duration = float(rospy.myargv()[1]) if len(rospy.myargv()) > 1 else 10
    rospy.Timer(rospy.Duration(duration), timer_callback, oneshot=True)
    rospy.spin()
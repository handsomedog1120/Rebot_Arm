#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebot_ponits_sim_node — RViz 末端轨迹记录节点(仿真用).

行为(配合 UI「◎ 移动到位置」按钮):
  1. 平时不记录, 不追加任何轨迹点(RViz 中保留上一次的轨迹).
  2. UI 点「移动到位置」(手动输入/画框两条路径) 时会在 /arm_trail_record
     发一帧 std_msgs/Bool(data=True); 本节点收到后:
       - 先 DELETEALL 清掉旧轨迹
       - 以触发瞬间的 gripper_end 位置为起点, 按 10Hz 采样末端 TF
       - LINE_STRIP 持续发布到 /arm_tip_marker
  3. 自动停止(=机械臂到位): 先确认关节确实动过(相对触发点 >MOVE_EPS),
     随后连续 0.5s 每拍关节变化 <SETTLE_EPS 即判定到位, 停止追加,
     最后再发一帧保证轨迹留在 RViz.
  4. 保护: 触发后 6s 内没检测到运动(误触发) -> 清除本次记录;
     单次记录超过 60s 强制结束.

说明:
  - 关节状态由 UI 的 30Hz 定时器持续发到 /joint_states, robot_state_publisher
    据此发布 base_link->...->gripper_end 的 TF, 本节点只订阅话题/查 TF,
    不再 import rebot_arm_server / motorbridge(那是 CAN 硬件后端, 独立进程里
    _ctrl/pos1 永远是 None, 旧版本节点因此一启动就崩), 系统 noetic python3 即可运行.
  - 本节点不再发布 /joint_states, 避免与 UI 两个发布者互相覆盖.
"""
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

BASE_FRAME = 'base_link'
TIP_FRAME = 'gripper_end'

RATE_HZ = 10.0
MOVE_EPS = 0.03          # 相对触发时刻任一关节变化 > 此值(rad)才算"动了"(约1.7°)
SETTLE_EPS = 0.005       # 每拍(0.1s)关节最大变化 < 此值(rad)视为静止
SETTLE_TICKS = 5         # 连续静止拍数: 10Hz x 5 = 0.5s -> 判定到位
NO_MOTION_TICKS = 60     # 触发后 6s 还没动 -> 放弃(误触发)
MAX_SESSION_TICKS = 600  # 单次记录最长 60s 兜底


class TrailRecorder(object):
    def __init__(self):
        self.q = None                 # 最近一帧 6 关节角(rad)
        self.recording = False
        self.q_start = None          # 触发时刻关节角
        self.q_last = None           # 上一拍关节角
        self.motion_seen = False
        self.settle_cnt = 0
        self.ticks = 0
        self.points = []

        rospy.init_node('rebot_points_sim_node')
        self.marker_pub = rospy.Publisher('/arm_tip_marker', Marker, queue_size=10)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lr = tf2_ros.TransformListener(self.tf_buf)
        rospy.Subscriber('/joint_states', JointState, self._js_cb, queue_size=10)
        rospy.Subscriber('/arm_trail_record', Bool, self._trig_cb, queue_size=4)
        rospy.loginfo('[trail] rebot_ponits_sim_node 就绪, 等待 UI「移动到位置」触发...')

    # ---------------- ROS 回调 ----------------
    @staticmethod
    def _arm_q(msg):
        """从 JointState 取 joint1..6; UI 发6关节, 兼容带夹爪的8关节消息."""
        names = list(msg.name)
        try:
            return [float(msg.position[names.index('joint%d' % k)]) for k in range(1, 7)]
        except ValueError:
            pos = list(msg.position)
            return [float(v) for v in pos[:6]] if len(pos) >= 6 else None

    def _js_cb(self, msg):
        q = self._arm_q(msg)
        if q is not None and len(q) == 6:
            self.q = q

    def _trig_cb(self, msg):
        """UI 点「移动到位置」: 开始一次新记录."""
        if not msg.data:
            return
        self._clear_marker()
        p0 = self._tip_point()       # 起点 = 触发瞬间末端位置
        self.points = [p0] if p0 is not None else []
        self.q_start = list(self.q) if self.q is not None else None
        self.q_last = list(self.q) if self.q is not None else None
        self.motion_seen = False
        self.settle_cnt = 0
        self.ticks = 0
        self.recording = True
        rospy.loginfo('[trail] 收到移动触发, 待运动确认后开始采点...')

    # ---------------- 标记/TF ----------------
    def _clear_marker(self):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.ns = 'trail'
        m.action = Marker.DELETEALL
        self.marker_pub.publish(m)

    def _tip_point(self):
        try:
            tf = self.tf_buf.lookup_transform(BASE_FRAME, TIP_FRAME, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        return Point(x=t.x, y=t.y, z=t.z)

    def _publish_line(self):
        if not self.points:
            return
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = rospy.Time.now()
        m.ns = 'trail'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.005                      # 线宽(米), LINE_STRIP 只看 scale.x
        m.color.r = 1.0
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        m.points = list(self.points)
        self.marker_pub.publish(m)

    # ---------------- 主循环 ----------------
    def spin(self):
        rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown():
            if self.recording:
                self.ticks += 1
                q = list(self.q) if self.q is not None else None

                if q is not None:
                    if not self.motion_seen:
                        # 阶段1: 等关节真正动起来(过滤手滑误触/目标不可达)
                        if self.q_start is not None and \
                                max(abs(a - b) for a, b in zip(q, self.q_start)) > MOVE_EPS:
                            self.motion_seen = True
                            rospy.loginfo('[trail] 检测到机械臂运动, 开始采点...')
                    else:
                        # 阶段2: 运动中采点; 连续静止 SETTLE_TICKS 拍 = 到位
                        dq = max(abs(a - b) for a, b in zip(q, self.q_last))
                        self.settle_cnt = self.settle_cnt + 1 if dq < SETTLE_EPS else 0

                        p = self._tip_point()
                        if p is not None:
                            self.points.append(p)
                        self._publish_line()

                        if self.settle_cnt >= SETTLE_TICKS:
                            self.recording = False
                            p_end = self._tip_point()
                            if p_end is not None:
                                self.points.append(p_end)
                            self._publish_line()   # 静止前最后一帧, 轨迹保留在 RViz
                            rospy.loginfo('[trail] 机械臂已到位, 停止记录(共 %d 点)'
                                          % len(self.points))
                    self.q_last = q

                # 超时保护: 一直没动(误触发) -> 清掉; 单次超长 -> 收尾
                if self.recording and (
                        (not self.motion_seen and self.ticks >= NO_MOTION_TICKS)
                        or self.ticks >= MAX_SESSION_TICKS):
                    why = '触发后 6s 无运动, 放弃记录' if not self.motion_seen \
                        else '单次记录超 60s, 强制结束'
                    self.recording = False
                    if not self.motion_seen:
                        self._clear_marker()
                        self.points = []
                    else:
                        self._publish_line()
                    rospy.loginfo('[trail] %s' % why)

            rate.sleep()


def main():
    TrailRecorder().spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

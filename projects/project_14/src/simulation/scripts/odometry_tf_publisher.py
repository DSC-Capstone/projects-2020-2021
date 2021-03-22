#!/usr/bin/env python

import rospy
import tf2_ros

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped


class OdomTfPublisher:
    def __init__(self):
        # Initalize anonoymous node
        rospy.init_node("cmd_vel_to_ackermann_drive", anonymous=True)
        self.odom_broadcaster = tf2_ros.TransformBroadcaster()

        odom_topic = rospy.get_param("~odom_topic", "/odom")

        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

        rospy.logwarn("Odometry Transform Publisher Created")

    def odom_callback(self, data):
        # rospy.logwarn(data)
        p = (data.pose.pose.position.x, data.pose.pose.position.y, 0)
        q = (data.pose.pose.orientation.x,
                     data.pose.pose.orientation.y,
                     data.pose.pose.orientation.z,
                     data.pose.pose.orientation.w)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = p[0]
        t.transform.translation.y = p[1]
        t.transform.translation.z = 0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        t2 = TransformStamped()
        t2.header.stamp = rospy.Time.now()
        t2.header.frame_id = "map"
        t2.child_frame_id = "odom"
        t2.transform.translation.x = p[0]
        t2.transform.translation.y = p[1]
        t2.transform.translation.z = 0
        t2.transform.rotation.x = q[0]
        t2.transform.rotation.y = q[1]
        t2.transform.rotation.z = q[2]
        t2.transform.rotation.w = q[3]

        self.odom_broadcaster.sendTransform(t)
        self.odom_broadcaster.sendTransform(t2)

        # self.odom_broadcaster.sendTransform(
        #     odom_pos,
        #     odom_quat,
        #     rospy.Time.now(),
        #     "map",
        #     "base_link",
        # )


if __name__ == "__main__":
    try:
        OdomTfPublisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

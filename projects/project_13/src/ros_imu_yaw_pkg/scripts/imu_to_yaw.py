#!/usr/bin/env python

import rospy
import math

from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion


class IMUYaw:
    def __init__(self):
        rospy.init_node("razor_yaw_node")
        yaw_topic = rospy.get_param("~yaw_topic", "/razor/yaw")
        imu_topic = rospy.get_param("~imu_topic", "/razor/imu")

        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback)
        self.yaw_pub = rospy.Publisher(yaw_topic, Float64, queue_size=1)

    def imu_callback(self, data):
        q = (data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w
        )

        _, _, yaw = euler_from_quaternion(q)

        yaw_deg = yaw * 180 / math.pi

        self.yaw_pub.publish(yaw_deg)

if __name__ == "__main__":
    imu_yaw_converter = IMUYaw()
    rospy.spin()

#!/usr/bin/env python
import pandas as pd
import rospy
from nav_msgs.msg import Odometry


class WaypointLogger:
    def __init__(self):
        self.output_name = rospy.get_param("~wp_log_output", "saved_waypoints.csv")
        self.odom_topic = rospy.get_param("~odom_topic", "/vehicle/odom")

        # save waypoints from odometry in csv
        self.wps = []

        self.current_count = 0
        self.logging_time = 100

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        rospy.loginfo("Started Odometry")

    def odom_callback(self, data):
        waypoint = (data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z,
                    data.pose.pose.orientation.x,
                    data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z,
                    data.pose.pose.orientation.w)
        self.wps.append(waypoint)

        # Save waypoint logs every self.logging_time iterations
        if self.current_count % self.logging_time == 0:
            rospy.logerr(f"saving odometry logs to {self.output_name}")
            self.df_logs = pd.DataFrame(self.wps, columns=["x", "y", "z", "qx", "qy", "qz", "qw"])
            self.df_logs.to_csv(self.output_name, index=False)

        self.current_count += 1


if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_logger', anonymous=True)
        WaypointLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

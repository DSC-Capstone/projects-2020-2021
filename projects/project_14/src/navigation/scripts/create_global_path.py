#!/usr/bin/env python
import pandas as pd
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GlobalPathCreator:
    def __init__(self):
        self.global_path_topic = "/global_path"
        self.global_path_csv = rospy.get_param("~global_path_csv", "global_path.csv")

        self.global_path = Path()
        self.global_path.header.frame_id = "/map"
        self.global_path.header.stamp = rospy.Time.now()

        # create path publisher
        self.path_pub = rospy.Publisher(self.global_path_topic, Path, queue_size=1)

        self.create_path()

        while not rospy.is_shutdown():
            self.path_pub.publish(self.global_path)

    def create_path(self):
        waypoints = self.get_waypoints(self.global_path_csv)
        self.prepare_waypoint_poses(waypoints)

    def get_waypoints(self, waypoints_csv):
        """
        Establishes waypoints from csv of sensor readings
        ----------------
        Params: csv_file
        csv file of data
        ----------------
        Returns: waypoints filtered from csv_file as a dataframe
        """

        data = pd.read_csv(waypoints_csv)
        data = data[280:-180]

        # get waypoints every 15 datapts
        data = data.iloc[::15, :]

        # add the starting point to end to form loop
        path_15 = data.append(data.iloc[0], ignore_index=True)

        return path_15

    def make_pose(self, row):
        """
        Makes Pose() with position and orientation
        ----------------
        Params: row
        odom reading at a single point in time
        ----------------
        Returns: Pose() object
        """
        pose = PoseStamped()
        pose.header.frame_id = "/map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = row.x
        pose.pose.position.y = row.y
        pose.pose.position.z = row.z
        pose.pose.orientation.x = row.qx
        pose.pose.orientation.y = row.qy
        pose.pose.orientation.z = row.qz
        pose.pose.orientation.w = row.qw

        return pose

    def prepare_waypoint_poses(self, waypoints: pd.DataFrame):
        """
        Makes PoseArray() with waypoint position & orientation
        ----------------
        Params: waypoints
        dataset of waypoints
        ----------------
        Returns: None
        """
        for i in range(len(waypoints)):
            pose = self.make_pose(waypoints.iloc[i])
            self.global_path.poses.append( pose )



if __name__ == "__main__":
    try:
        rospy.init_node("global_path_creator", anonymous=True)
        GlobalPathCreator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

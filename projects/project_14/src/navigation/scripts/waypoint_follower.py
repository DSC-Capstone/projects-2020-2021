#!/usr/bin/env python

import math
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Path
from geometry_msgs.msg import Quaternion, Pose, Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion

def euclidean_dist(x1, x2, y1, y2):
    """
    Calculates Euclidean distance.
    -------------
    Params: xy coordinates
    -------------
    Returns: distance
    """
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

class FollowWaypoints:
    def __init__(self):
        # Initalize anonoymous node
        rospy.init_node("waypoint_follower", anonymous=True)

        self.ackermann_cmd_topic = rospy.get_param("~ackermann_cmd_topic", "/ackermann_cmd")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/global_path")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")

        self.update_rate = rospy.get_param("~update_rate", 10)
        self.rate = rospy.Rate(self.update_rate)

        # self.curr_x = rospy.get_param("~starting_x", 0)
        # self.curr_y = rospy.get_param("~starting_y", 0)
        self.goal_x = rospy.get_param("~starting_x", 0)
        self.goal_y = rospy.get_param("~starting_y", 0)
        self.global_path = None
        self.curr_pos = None
        self.curr_heading = None
        self.max_steering_angle = 0.34

        self.K_psteer = 1
        self.K_dsteer = 0.25
        self.K_ppath = 0.25
        self.K_dpath = 0.25

        rospy.Subscriber(self.global_path_topic, Path, self.global_path_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        self.pub_drive = rospy.Publisher(self.ackermann_cmd_topic,
                                         AckermannDriveStamped,
                                         queue_size=1)


        # don't navigate until we have a path, curernt postion and the heading
        while self.global_path is None or self.curr_pos is None or self.curr_heading is None:
            continue

        self.navigate_to_waypoint_x()

    def odom_callback(self, data):
        quat_heading = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        self.curr_heading = euler_from_quaternion(quat_heading)
        self.curr_pos = data.pose


    def global_path_callback(self, data):
        self.global_path = data

    def _publish_cmd_vel(self):
        """Continuously publish ackermann command to vehicle"""


    def get_steering(self, curr_pos, waypoint, curr_heading):
        """
        Calculates steering angle to point
        towards next waypoint
        -------------
        Params:
        curr_pos: current vehicle position in x, y
        waypoint: upcoming waypoint pose()
        curr_heading: current vehicle heading in (degrees/radians?)
        -------------
        Returns:
        Steering angle
        """
        # angle between waypoint and current position in radians
        desired_heading = math.atan(
            (waypoint.pose.position.y - curr_pos.pose.position.y) /
            (waypoint.pose.position.x - curr_pos.pose.position.x))

        if curr_heading != desired_heading:
            #rotate steering_angle degrees
            return desired_heading
        return

    def navigate_to_waypoint_x(self):
        """
        Handles waypoint selection and waypoint navigation
        -------------
        Params:
        curr_pos: current vehicle position as point object
        curr_heading: current vehicle heading in (degrees/radians?)
        path: waypoints as Path()
        """

        amt_waypoints = len(self.global_path.poses)

        # waypt radius buffer in ?? units
        # choose CEP? but make sure waypoints are far away enough from
        # each other to be greather than CEP
        buffer_radius = 0.15

        self.prev_waypt = None
        for i in range(amt_waypoints):
            # number of points to lookahead needs to be based off of distance
            # between points
            lookahead_pt = self.global_path.poses[(i + 2) % amt_waypoints]
            curr_waypt = self.global_path.poses[i]

            if self.prev_waypt is None:
                self.prev_waypt = curr_waypt

            # Calculate the distance between the current waypoint and endpoint.
            dist_waypt_to_end = euclidean_dist(lookahead_pt.pose.position.x,
                                               curr_waypt.pose.position.x,
                                               lookahead_pt.pose.position.y,
                                               curr_waypt.pose.position.y)
            # Calculate the distance between the current waypoint and current vehicle position.
            dist_curr_to_end = euclidean_dist(lookahead_pt.pose.position.x,
                                              self.curr_pos.pose.position.x,
                                              lookahead_pt.pose.position.y,
                                              self.curr_pos.pose.position.y)

            dist_curr_to_waypt = euclidean_dist(curr_waypt.pose.position.x,
                                                self.curr_pos.pose.position.x,
                                                curr_waypt.pose.position.y,
                                                self.curr_pos.pose.position.y)

            dist_curr_to_prev = euclidean_dist(self.prev_waypt.pose.position.x,
                                                self.curr_pos.pose.position.x,
                                                self.prev_waypt.pose.position.y,
                                                self.curr_pos.pose.position.y)

            dist_prev_to_waypt = euclidean_dist(self.prev_waypt.pose.position.x,
                                                curr_waypt.pose.position.x,
                                                self.prev_waypt.pose.position.y,
                                                curr_waypt.pose.position.y)

            rospy.logwarn(f"[{i}]: {dist_waypt_to_end} -- {dist_curr_to_end}")

            # continue towards curr_waypt
            if dist_waypt_to_end < dist_curr_to_end:
                rospy.logwarn(" heading towards curr waypoint ")

                curr_wypt_orientation_q = (
                    curr_waypt.pose.orientation.x,
                    curr_waypt.pose.orientation.y,
                    curr_waypt.pose.orientation.z,
                    curr_waypt.pose.orientation.w
                )

                _, _, waypoint_heading = euler_from_quaternion(curr_wypt_orientation_q)
                self.heading_error_prev = 0
                self.path_error_prev = 0

                while dist_curr_to_waypt > buffer_radius:
                    drive = AckermannDriveStamped()
                    curr_orientation_q = (
                        self.curr_pos.pose.orientation.x,
                        self.curr_pos.pose.orientation.y,
                        self.curr_pos.pose.orientation.z,
                        self.curr_pos.pose.orientation.w
                    )
                    _, _, current_heading = euler_from_quaternion(curr_orientation_q)

                    # desired_heading = waypoint_heading
                    desired_heading = math.atan(
                        (self.curr_pos.pose.position.y - curr_waypt.pose.position.y) /
                        (self.curr_pos.pose.position.x - curr_waypt.pose.position.x))
                    self.heading_error = desired_heading - current_heading
                    if self.heading_error > math.pi / 2:
                        self.heading_error -= math.pi
                    elif self.heading_error < - math.pi / 2:
                        self.heading_error += math.pi
                    heading_error_delta = self.heading_error_prev - self.heading_error

                    # path error
                    a = dist_prev_to_waypt
                    b = dist_curr_to_waypt
                    c = dist_curr_to_prev
                    theta_1 = math.acos((c**2 + a**2 - b**2) / (2 * c * a))

                    vec_1 = (curr_waypt.pose.position.x - self.prev_waypt.pose.position.x,
                             curr_waypt.pose.position.y - self.prev_waypt.pose.position.y)
                    vec_2 = (self.curr_pos.pose.position.x - self.prev_waypt.pose.position.x,
                             self.curr_pos.pose.position.y - self.prev_waypt.pose.position.y)
                    steering_angle_sign = vec_1[0] * vec_2[1] - vec_2[0] * vec_1[1]

                    self.path_error = c * math.sin(theta_1) * steering_angle_sign * -1
                    path_error_delta = self.path_error_prev - self.path_error

                    # rospy.logwarn(dist_curr_to_waypt)

                    # rospy.logwarn(f"{self.heading_error} -- {heading_error_delta} -- \
                    #                 {self.path_error} -- {path_error_delta}")

                    steering_angle = self.K_psteer * self.heading_error \
                                     + self.K_dsteer * heading_error_delta \
                                     + self.K_ppath * self.path_error \
                                     + self.K_dpath * path_error_delta

                    # constrain to maximum and minimum steering angles of +-20 degreees
                    steering_angle = min(max(steering_angle, -0.34), 0.34)
                    # rospy.logwarn(steering_angle)

                    # Fill drive message
                    drive.drive.steering_angle = steering_angle
                    drive.drive.speed = .5

                    # send steering and throttle to car
                    self.pub_drive.publish(drive)
                    self.heading_error_prev = self.heading_error
                    self.path_error_prev = self.path_error

                    rospy.logwarn(f"{dist_curr_to_waypt} -- {desired_heading} -- {steering_angle}")

                    # continuously check if current_pos is within buffer radius
                    # should be getting new curr_pos updates here
                    dist_curr_to_waypt = euclidean_dist(curr_waypt.pose.position.x,
                                                        self.curr_pos.pose.position.x,
                                                        curr_waypt.pose.position.y,
                                                        self.curr_pos.pose.position.y)
                    dist_curr_to_prev = euclidean_dist(self.prev_waypt.pose.position.x,
                                                        self.curr_pos.pose.position.x,
                                                        self.prev_waypt.pose.position.y,
                                                        self.curr_pos.pose.position.y)

                rospy.logwarn("within buffer radius, head towards next one")

            rospy.logwarn("=" * 80)
            rospy.logwarn("Curr_waypoint already reached, heading to next one")
            rospy.logwarn("=" * 80)

if __name__ == "__main__":
    try:
        FollowWaypoints()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

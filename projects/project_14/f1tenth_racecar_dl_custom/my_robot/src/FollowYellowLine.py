#!/usr/bin/env python

import rospy
import numpy as np
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class YellowLineFollower:

  SPEED = 0.6
  QUEUE_SIZE = 1
  WHEELBASE = 0.33

  def __init__(self):
    self.bridge_object = CvBridge()
    self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                      Image, self.camera_callback)
    # self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/teleop',
    #                                    AckermannDriveStamped, queue_size= self.QUEUE_SIZE)
    self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',
                                       AckermannDriveStamped, queue_size= self.QUEUE_SIZE)
    # /vesc/low_level/ackermann_cmd_mux/output 
    self.last_angle = None
    self.test1_pub = rospy.Publisher('/test/res_raw', Image, queue_size = self.QUEUE_SIZE)
    self.test2_pub = rospy.Publisher('/test/hls_raw', Image, queue_size = self.QUEUE_SIZE)
    self.test3_pub = rospy.Publisher('test/mask_raw', Image, queue_size = self.QUEUE_SIZE)

  def convert_angular_velocity_to_steering_angle(self, angular_velocity):
      if angular_velocity == 0:
        return 0
      return math.atan(angular_velocity * (self.WHEELBASE/self.SPEED))
    
  def camera_callback(self, data):

    try:
        cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding='bgr8')
    except CvBridgeError as e:
        print(e)

    height, width, channels = cv_image.shape   
    top_trunc = int(height/2)
    mid_width = width/2
    margin = width/4
    left_bound = int(mid_width - margin)
    right_bound = int(mid_width + margin)
    crop_img = cv_image[top_trunc: ,0:width] 
    # image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
    # hls = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)#.astype(np.float)

    # lower_yellow = np.array([ 0, 105,  175])
    # upper_yellow = np.array([29, 255, 255])
    # mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    hls = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float)
    lower_yellow = np.array([171, 135, 82])
    upper_yellow = np.array([255, 255, 125])
    mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    
    # rows_to_watch = 100
    # top_trunc = int(height/2)
    # bot_trunc = int(top_trunc + rows_to_watch)
    # crop_img = cv_image[top_trunc:bot_trunc, 0:width]
    m = cv2.moments(mask, False)
    try:
        cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
    except ZeroDivisionError:
        cx, cy = width / 2, height / 2
    res = cv2.bitwise_and(crop_img, crop_img, mask = mask)
    cv2.circle(res, (int(cx), int(cy)), 10, (0, 0, 255), -1)
#    cv2.imshow("Original", cv_image)
#    cv2.imshow("HSV", hls)
#    cv2.imshow("MASK", mask)
#    cv2.imshow("RES", res)
#    cv2.waitKey(0)
    error_x = cx - width/2
    angular_z = -error_x/100
    if angular_z == 0:
        steering_angle = self.last_angle
    else:
        steering_angle = self.convert_angular_velocity_to_steering_angle(angular_z)
        self.last_angle = steering_angle
    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = data.header.stamp
    drive_msg.drive.speed = self.SPEED
    steering_angle = np.clip(steering_angle, -0.4, 0.4)
    drive_msg.drive.steering_angle = steering_angle
    self.drive_pub.publish(drive_msg)
    self.test2_pub.publish(self.bridge_object.cv2_to_imgmsg(hls))
    self.test3_pub.publish(self.bridge_object.cv2_to_imgmsg(mask))
    self.test1_pub.publish(self.bridge_object.cv2_to_imgmsg(res))


if __name__ == '__main__':
  rospy.init_node('dsc_190_team_1', anonymous = True)
  robot = YellowLineFollower()
  rospy.sleep(0.1)
  rospy.spin()

  def shutdownhook():
      cv2.destroyAllWindows()

  rospy.on_shutdown(shutdownhook)


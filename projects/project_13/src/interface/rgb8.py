import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rosbag 
import cv2

path_to_bag = '/home/SENSETIME/shijia/Desktop/data.bag'
topic='/cmr_filtered/image'

times = []
images = []

bridge = CvBridge()

with rosbag.Bag(path_to_bag) as bag:
    
    topics = bag.get_type_and_topic_info().topics
    if topic not in topics:
        raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

    for topic, msg, t in bag.read_messages(topics=[topic]):
        time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
        times.append(time)

        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        import matplotlib.pyplot as plt
        cv2.imwrite('camera.png',img)
        images.append(img) 
        import time
        time.sleep(0.5)

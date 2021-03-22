#!/usr/bin/env python
import pandas as pd
import rospy
from sensor_msgs.msg import NavSatFix


class GPSLogger:
    def __init__(self):
        rospy.init_node("gps_logger", anonymous=True)
        self.sub = rospy.Subscriber('/ublox/fix', NavSatFix, self.gps_callback)
        self.logs = []
        rospy.on_shutdown(self.shutdown_hook)
        rospy.spin()

    def gps_callback(self, data):
        lat = data.latitude
        lon = data.longitude
        alt = data.altitude
        rospy.loginfo("GPS: (%f, %f, %f)", lat, lon, alt)
        self.logs.append((lat, lon, alt))

    def shutdown_hook(self):
        rospy.loginfo("Saving Logs")
        df_logs = pd.DataFrame(self.logs, columns=["lat", "lon", "alt"])
        df_logs.to_csv("zed_f9p_fixed.csv", index=False)



if __name__ == "__main__":
    try:
        gl = GPSLogger()
    except rospy.ROSInterruptException:
        rospy.loginfo("Stopping gps_logger node")
        pass

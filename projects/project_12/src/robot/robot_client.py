from datetime import datetime
from pymavlink import mavutil


class RobotClient:
    def __init__(self, **robot_cfg):
        self.url = robot_cfg["url"]
        self.mavcar = mavutil.mavlink_connection(self.url)
        self.save_logs = robot_cfg["save_logs"]
        self.debug = robot_cfg["debug"]
        self.log_file = robot_cfg["log_path"] + \
            datetime.now().strftime("%d-%m-%Y_%H-%M") + ".log"
        self.closed = False

    def get_gps(self) -> None:
        """Get GPS coordinates from connected vehicle."""
        while not self.closed:
            coords = self.mavcar.location()
            if self.save_logs:
                self.append_log(coords)

    def append_log(self, log: mavutil.location) -> None:
        """Append received log to text file."""
        with open(self.log_file, 'a') as f:
            if self.debug:
                print(log)
            f.write(f"{str(log)}\n")


if __name__ == "__main__":
    url = "tcp:127.0.0.1:5760"
    robot = RobotClient(url)
    robot.get_gps()

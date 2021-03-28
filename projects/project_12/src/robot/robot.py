import subprocess

class Robot:
    def __init__(self, **robot_cfg):
        try:
            subprocess.run([robot_cfg["command"],
                            robot_cfg["vehicle_type"]])
        except KeyboardInterrupt:
            print("Shutting down SITL")

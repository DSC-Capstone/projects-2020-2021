import os
import shutil
import bagpy
import rosbag
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import math
import numpy as np
from .utils.Quat_Euler import euler_from_quaternion

# IMU Mount Analysis Plots

def mounting_imu_plot(filename, outdir):
    df_imu = pd.read_csv(filename)
    df_imu['Time'] - df_imu['Time'].min()

    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['linear_acceleration.x'], label='Lin. Accel X')
    plt.plot(df_imu['Time'], df_imu['linear_acceleration.y'], label='Lin. Accel Y')
    plt.axhline(y=0, color='black', xmin=0.04, xmax=0.96, label= 'Zero Accel')
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Linear Acceleration (m/$s^2$)', fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Linear Acceleration', fontsize=20)

    name = filename.split("/")[-1]
    name = name.split(".")[0]

    plt.savefig(os.path.join(outdir, "Mounting_" + name + ".png"))
    print("Mounting " + name + ' plot success!')

def mounting_yaw_plot(filename, outdir):

    df_imu = pd.read_csv(filename)
    df_imu['Time'] = df_imu['Time'] - df_imu['Time'].min()

    quat = np.array([df_imu['orientation.x'], 
                 df_imu['orientation.y'], 
                 df_imu['orientation.y'], 
                 df_imu['orientation.z']])

    quat = quat.reshape(len(quat[0]), 4)

    temp1 = [euler_from_quaternion(arr[0], arr[1], arr[2], arr[3]) for arr in quat]

    yaw_array = [t[2] for t in temp1]
    yaw_deg = [x*180/math.pi for x in yaw_array]

    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], yaw_deg)
    plt.axhline(y=0, color='red', xmin=0.04, xmax=0.96)
    plt.title('Yaw', fontsize=20)

    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Yaw (deg)', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    name = filename.split("/")[-1]
    name = name.split(".")[0]

    plt.savefig(os.path.join(outdir, "Mounting_Yaw_" + name + ".png"))
    print("Mounting Yaw " + name + ' plot success!')




# IMU Half Arc Analysis Plots

def half_arc_yaw_plot(filename, plot_range, outdir):
    df_imu = pd.read_csv(filename)
    df_imu['Time'] - df_imu['Time'].min()
    yaw_offset = [x + 360 if x < 0 else x for x in df_imu['data']]
    plt.figure(figsize=(10,8))
    plt.title('West to East Yaw', fontsize=20)
    plt.plot(df_imu['Time'][:plot_range], yaw_offset[:plot_range])
    plt.axhline(y=270, color='red', xmin=0.04, xmax=0.96, label= 'Zero Accel', linestyle='dashed')
    plt.axhline(y=90, color='red', xmin=0.04, xmax=0.96, label= 'Zero Accel', linestyle='dashed')
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Yaw (deg)', fontsize=20)

    name = filename.split("/")[-1]
    name = name.split(".")[0]

    plt.savefig(os.path.join(outdir,  "Half_Arc_" + name + ".png"))
    print("Half Arc IMU " + name + ' plot success!')

# IMU Straight Line Analysis Plots
def straight_line_yaw_plot(filename, outdir):
    df_imu = pd.read_csv(filename)
    df_imu['Time'] - df_imu['Time'].min()

    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['data'])

    plt.axhline(y=90, color='red', xmin=0.04, xmax=0.96)

    plt.title('Straight Line Yaw Test', fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Yaw (deg)', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    name = filename.split("/")[-1]
    name = name.split(".")[0]

    plt.savefig(os.path.join(outdir,  "Straight_Line_Yaw_" + name + ".png"))
    print("Straight Line Yaw " + name + ' plot success!')

# IMU topic data plots first
def report_plots(filename, outdir):
    df_imu = pd.read_csv(filename)
    df_imu['Time'] - df_imu['Time'].min()
    
    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['linear_acceleration.x'])
    plt.title('IMU X-Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.savefig(os.path.join(outdir, 'accel_x_plt.png'))
    print('X-acceleration plot success!')
    
    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['linear_acceleration.y'])
    plt.title('IMU Y-Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.savefig(os.path.join(outdir, 'accel_y_plot.png'))
    print('Y-acceleration plot success!')
    
    plt.figure(figsize=(10,8))
    plt.scatter(df_imu['linear_acceleration.x'], df_imu['linear_acceleration.y'], alpha=0.5)
    plt.title('IMU Scatterplot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(outdir, 'accel_scatter_plot.png'))
    print('scatter-acceleration plot success!')
    
    plt.figure(figsize=(10,8))
    plt.hist(df_imu['linear_acceleration.x'], bins=None)
    plt.title('IMU X-Acceleration Distribution')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(outdir, 'accel_x_hist.png'))
    print('X-acceleration hist plot written successfully!')
    
    plt.figure(figsize=(10,8))
    plt.hist(df_imu['linear_acceleration.y'], bins=None)
    plt.title('IMU X-Acceleration Distribution')
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(outdir, 'accel_y_hist.png'))
    print('Y-acceleration hist plot success!')

    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['orientation.x'])
    plt.title('Orientation X-Value')
    plt.xlabel('Time')
    plt.ylabel('Quaternion (X)')
    plt.savefig(os.path.join(outdir, 'orient_x_plot.png'))
    print('Orientation x plot success!')
   
    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['orientation.y'])
    plt.title('Orientation Y-Value')
    plt.xlabel('Time')
    plt.ylabel('Quaternion (X)')
    plt.savefig(os.path.join(outdir, 'orient_y_plot.png'))
    print('Orientation y plot success!')
    
    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['orientation.z'])
    plt.title('Orientation Z-Value')
    plt.xlabel('Time')
    plt.ylabel('Quaternion (Z)')
    plt.savefig(os.path.join(outdir, 'orient_z_plot.png'))
    print('Orientation z plot success!')

    plt.figure(figsize=(10,8))
    plt.plot(df_imu['Time'], df_imu['orientation.z'])
    plt.title('Orientation W-Value')
    plt.xlabel('Time')
    plt.ylabel('Quaternion (W)')
    plt.savefig(os.path.join(outdir, 'orient_w_plot.png'))
    print('Orientation w plot success!')

def odom_plots(filename, outdir):

    df_odom = pd.read_csv(filename)
    
    name = filename.split("/")[-1]
    name = name.split("_")[-1]
    name = name.split(".")[0]

    plt.figure(figsize=(10,8))
    plt.plot(df_odom['pose.pose.position.x'], [0] * len(df_odom))
    plt.xlabel('Distance in meters')
    plt.title("ERPM_Gain " + name)
    plt.xticks(np.arange(0, 2.5 ,0.2))

    plt.savefig(os.path.join(outdir, 'ERPM_Gain_' + name + '.png'))
    print('ERPM_Gain ' + name + ' plot success!')

def odom_turn_plots(filename, outdir):
    df_odom = pd.read_csv(filename)

    name = filename.split("/")[-1]
    name = name.split("_")[-1]
    name = name.split(".")[0]

    plt.figure(figsize=(10,8))
    plt.plot(df_odom['pose.pose.position.x']-df_odom['pose.pose.position.x'].min(), 
        df_odom['pose.pose.position.y'],'b')
    plt.xlabel('X in meters')
    plt.ylabel('Y in meters')
    plt.title("Servo_to_ERPM_Gain " + name)
    
    plt.savefig(os.path.join(outdir, 'Servo_to_ERPM_Gain_' + name + '.png'))
    print('Servo_to_ERPM_Gain ' + name + ' plot success!')


def plot_all(outdir, IMU, Odom):

    # Reset outdir
    if (os.path.exists(outdir) and os.path.isdir(outdir)):
        shutil.rmtree(outdir)
    
    os.mkdir(outdir)

    #Plot IMU Mounting
    for filename in IMU['Mounting']:
        mounting_imu_plot(filename, outdir)
        mounting_yaw_plot(filename, outdir)

    #Plot IMU Half Arc
    for d in IMU['Half_Arc']:
        half_arc_yaw_plot(d["f"], d["r"], outdir)

    for filename in IMU['S_Line']:
        straight_line_yaw_plot(filename, outdir)

    #Plot IMU Report
    report_plots(IMU['Report'], outdir)

    #Plot Odom Tuning
    for filename in Odom['Vesc']:
        odom_plots(filename, outdir)

    #Plot Odom Arc Tuning
    for filename in Odom['Servo']:
        odom_turn_plots(filename, outdir)

    return


if __name__ == '__main__':
    main()


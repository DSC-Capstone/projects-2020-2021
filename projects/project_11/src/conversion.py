import os
import shutil
import bagpy
import rosbag
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import math
from glob import glob


def bagOpener(bag_path, topic_name):
	
	output_path = bag_path.replace('raw', 'clean').replace('bag', 'csv')

	split_path_arr = output_path.split("/")

	topic_path = split_path_arr[0] + "/" + split_path_arr[1] + "/" + split_path_arr[2]

	if not os.path.exists(topic_path):
		print(topic_path)
		os.mkdir(topic_path)

	b = bagreader(bag_path)
	data = b.message_by_topic(topic_name)
	df = pd.read_csv(data)
	df.to_csv(output_path)

	shutil.rmtree(bag_path.replace(".bag", ""))

	return


def convert(indir, outdir):
	
	if (os.path.exists(outdir) and os.path.isdir(outdir)):
		shutil.rmtree(outdir)
	
	os.mkdir(outdir)

	print('Output directory created successfully')

	data_raw_folders = glob(indir+"/*")

	for fold in data_raw_folders:
		
		name = fold.split("/")[-1]
		topic_name = ""

		if not os.path.isdir(fold):
			continue

		if name == 'vesc_odom':
			topic_name = '/vesc/odom'

		elif name == 'razor_yaw':
			topic_name = '/razor/yaw'

		elif name == 'razor_imu':
			topic_name = '/razor/imu'

		else:
			continue

		for f in glob(fold+"/*"):
			bagOpener(f, topic_name)


	print('Extracted .bag data and written to destination successfully')
	
	return


if __name__ == '__main__':
	main()

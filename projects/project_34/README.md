# Project: User Wait

## Introduction

Activity monitor in the computer visualizes the system performance, but we don't know when our system gets slow or halted. If we are able to predict the mouse wait time, users could terminate their processes ahead of time to avoid waiting. Currently, there is little research conducted on the mouse wait prediction.

## Running the project

- To get the data run the following command line located inside run.py file.
  This will call specific dll files to collect the data regard to your laptop.

- To get the data, from the project root dir, you should see the data folder.
  Inside the data folder, you should see a new database file being generated.

## Type Of Data

Data provided by Intel includes 14,534,433 rows with 29,587 unique GUID within the 2020 interval. Since the complete dataset is fairly large, we sample 1/14 of the dataset, which leaves 27,014 GUID for our model. For the target, we divide the wait time into 0-3s, 3-5s, 5-7s, and 7+s as a preparation for the classification model. After exploring the correlation between potential features and the mouse wait time, we find that dynamic features, including CPU utilization, disk utilization, hard page faults, and static features, including the number of cores, RAM, model type, etc, could influence the mouse wait time. These features are then used in the model.

The data set we use is ”mousewaitall.csv001”,which is provided by the Intel teams. This data setrecords kinds of system usage before and after mousewait happens. Each feature in this data set consists ofprefix, infix and suffix. Prefix has ”before” and ”after”.It represent is this feature recorded before or after mousewait event. Infix has ”CPUUtil”, ”harddpf”, ”diskutil”and ”networkutil”. This represents what kind of systemusage this feature records. Suffix has ”min”, ”max” and”mean”. This represents the way this feature computesstatistics.

The second data set we use is ”system sysinfo uniquenormalized.csv000”, which is provided by the Intel teams. It contain 32 different features and 100,000unique systems.This data set provides informationabout the system hardware like CPU model, GPU,ram, etc.

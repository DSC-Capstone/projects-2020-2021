
import pandas as pd
import gzip
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# First guessing: battery minutes remaining is related with number of devices
def num_dev_feature(battery_event, device_use):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    num_dev = device_use.set_index('guid').loc[set(battery_event.guid).intersection(device_use.guid)].reset_index().groupby(['guid']).name.count().sort_index()
    num_dev = num_dev.loc[set(data1.index).intersection(set(num_dev.index))]
    data1 = data1.loc[set(data1.index).intersection(set(num_dev.index))]

    print(np.corrcoef(data1, num_dev))

    
    return num_dev

# Second guessing: battery minutes remaining is related with number of process
def num_proc_feature(battery_event, process):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(battery_event.guid).intersection(set(process.guid))
    num_proc = process.set_index('guid').loc[needed].reset_index().groupby(['guid']).proc_name.count().sort_index()

    data1 = data1.loc[set(num_proc.index).intersection(data1.index)]
    num_proc = num_proc.loc[data1.index]

    print(np.corrcoef(data1, num_proc))
    return num_proc

# Third guessing: battery minutes remaining is related with Average Page Faults
def page_faults_feature(battery_event, process):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(battery_event.guid).intersection(set(process.guid))
    page_faults = process.set_index('guid').loc[needed].reset_index().groupby(['guid']).page_faults.mean().sort_index()

    data1 = data1.loc[set(page_faults.index).intersection(data1.index)]
    page_faults = page_faults.loc[data1.index]
    
    print(np.corrcoef(data1, page_faults))
    
    return page_faults

# Fourth guessing: battery minutes remaining is related with Average Memory
def avg_memory_feature(battery_event, process):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(battery_event.guid).intersection(set(process.guid))
    data2 = process.set_index('guid').loc[needed].reset_index().groupby(['guid']).avg_memory.mean().sort_index()

    data1 = data1.loc[set(data2.index).intersection(data1.index)]
    avg_memory = data2.loc[data1.index]
    
    print(np.corrcoef(data1, avg_memory))
    
    return avg_memory 

# Fifth guessing: battery minutes remaining is related with cpu_user_sec + cpu_kernel_sec
def cpu_sec_feature(battery_event, process):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(battery_event.guid).intersection(set(process.guid))
    process['cpu_sec']= process['cpu_user_sec']+process['cpu_kernel_sec']
    data2 = process.set_index('guid').loc[needed].reset_index().groupby(['guid']).cpu_sec.mean().sort_index()

    data1 = data1.loc[set(data2.index).intersection(data1.index)]
    cpu_sec = data2.loc[data1.index]
    
    print(np.corrcoef(data1, cpu_sec))
    
    return cpu_sec 

# Sixth guessing: battery minutes remaining is related with full_charge_capacity
def capacity_feature(battery_event, battery_info):
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()
    
    needed = set(battery_event.guid).intersection(set(battery_info.guid))
    data2 = battery_info.set_index('guid').loc[needed].reset_index().groupby(['guid']).full_charge_capacity.mean().sort_index()
    data1 = data1.loc[set(data2.index).intersection(data1.index)]
    capacity = data2.loc[data1.index]
    
    print(np.corrcoef(data1, capacity))
    
    return capacity 

# 7th guessing: battery minutes remaining is related with cpu_percent
def cpu_percent_feature(cpu, battery_event):
    cpu_info = cpu.groupby(['guid','name'])['mean'].mean().reset_index(level=[0,1])
    cpu_percent = cpu_info.loc[cpu_info.name == 'HW::CORE:C0:PERCENT:']
    
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(data1.index).intersection(set(cpu_percent.guid))
    cpu_percent = cpu_percent.set_index('guid').loc[needed]['mean']

    data1 = data1.loc[set(cpu_percent.index).intersection(data1.index)]
    cpu_percent = cpu_percent.loc[data1.index]
    
    print(np.corrcoef(data1, cpu_percent))
    
    return cpu_percent

# 8th guessing: battery minutes remaining is related with cpu_temperature
def cpu_temp_feature(cpu, battery_event):
    cpu_info = cpu.groupby(['guid','name'])['mean'].mean().reset_index(level=[0,1])
    cpu_centi_temp = cpu_info.loc[cpu_info.name == 'HW::CORE:TEMPERATURE:CENTIGRADE:']
    
    data1 = battery_event.groupby(['guid']).battery_minutes_remaining.mean()

    needed = set(data1.index).intersection(set(cpu_centi_temp.guid))
    cpu_temp = cpu_centi_temp.set_index('guid').loc[needed]['mean']

    data1 = data1.loc[set(cpu_temp.index).intersection(data1.index)]
    cpu_temp = cpu_temp.loc[data1.index]
    
    print(np.corrcoef(data1, cpu_temp))
    
    return cpu_temp
    
   

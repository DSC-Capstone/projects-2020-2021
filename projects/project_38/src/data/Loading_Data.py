import pandas as pd
import gzip
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Parameters
DATES = pd.date_range('2020-09', '2020-10', freq='D', closed='left')

DEVICE_INFP1 = 'devuse_4known_device.csv000.gz'
DEVICE_OUTFP1 = 'minimini_device_use1.csv'

DEVICE_INFP2 = 'devuse_4known_device.csv001.gz'
DEVICE_OUTFP2 = 'minimini_device_use2.csv'

BATTERY_EVENT_INFP = 'batt_acdc_events.csv000.gz'
BATTERY_EVENT_OUTFP = 'minimini_battery_event2.csv'

BATTERY_INFO_INFP = 'batt_info.csv000.gz'
BATTERY_INFO_OUTFP = 'minimini_battery_info2.csv'

PROCESS_INFP1 = 'plist_process_resource_util_13wks.csv000.gz'
PROCESS_OUTFP1 = 'minimini_process1.csv'


HW_INFP1 = 'hw_metric_histo.csv000.gz'
HW_OUTFP1 = 'minimini_hw1.csv'

HW_INFP2 = 'hw_metric_histo.csv001.gz'
HW_OUTFP2 = 'minimini_hw2.csv'
    

def load_device(DEVICE_OUTFP1, DEVICE_OUTFP2):
    
    df3 = pd.read_csv(DEVICE_OUTFP1, index_col = 0)
    df4 = pd.read_csv(DEVICE_OUTFP2, index_col = 0)
    device_use = df4.append(df3)
    newcol = ['dt', 'load_ts', 'batch_id', 'audit_zip', 'audit_internal_path', 'guid',
           'interval_start_utc', 'interval_end_utc', 'interval_local_start',
           'interval_local_end', 'ts','device', 'hw_name', 'name',
           'duration', 'status']
    device_use.columns = newcol

    return device_use
    
def load_battery_event(BATTERY_EVENT_OUTFP):
    
    battery_event = pd.read_csv(BATTERY_EVENT_OUTFP, index_col = 0)
    newcol = ['dt', 'guid','load_ts','batch_id','audit_zip','audit_internal_path',
           'interval_start_utc', 'interval_end_utc', 'interval_local_start',
           'interval_local_end', 'ts','system_power_state', 'event_type',
           'duration', 'battery_percent_remaining', 'battery_minutes_remaining']
    battery_event.columns = newcol
    
    return battery_event

def load_battery_info(BATTERY_INFO_OUTFP):
    
    battery_info = pd.read_csv(BATTERY_INFO_OUTFP, index_col = 0)
    newcol = ['dt', 'guid','load_ts','batch_id','audit_zip','audit_internal_path',
           'interval_start_utc', 'interval_end_utc', 'interval_local_start',
           'interval_local_end', 'ts','battery_enum', 'chemistry',
           'designed_capacity', 'full_charge_capacity', 'battery_count']
    battery_info.columns = newcol
    
    return battery_info

def load_process(PROCESS_OUTFP1):
    process = pd.read_csv(PROCESS_OUTFP1, index_col = 0)
    newcol = ['dt', 'guid','load_ts','batch_id','audit_zip','audit_internal_path','interval_start_utc', 
     'interval_end_utc', 'interval_local_start',
           'interval_local_end', 'ts','proc_name', 'exe_hash',
           'num_runs', 'ttl_run_tm_in_ms', 'cpu_user_sec', 'cpu_kernel_sec',
           'io_bytes_read', 'io_bytes_write', 'io_bytes_other', 'page_faults',
           'hard_page_faults', 'disk_read_iobytes', 'disk_write_iobytes',
           'tcpip_sendbytes', 'tcpip_receivebytes', 'udpip_sendbytes',
           'udpip_receivebytes', 'avg_memory', 'peak_memory']
    process.columns = newcol
    
    return process

def load_cpu(HW_OUTFP1, HW_OUTFP2):
    cpu1 = pd.read_csv(HW_OUTFP1, index_col = 0)
    cpu2 = pd.read_csv(HW_OUTFP2, index_col = 0)
    cpu = cpu1.append(cpu2)
    newcol = ['dt', 'guid','load_ts','batch_id','audit_zip','audit_internal_path',
           'interval_start_utc', 'interval_end_utc', 'interval_local_start',
           'interval_local_end', 'name', 'instance', 'nrs', 'mean',
           'histogram_min', 'histogram_max', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
           'bin_5', 'bin_6', 'bin_7', 'bin_8', 'bin_9', 'bin_10',
           'metric_max_val']
    cpu.columns = newcol
    
    return cpu
 

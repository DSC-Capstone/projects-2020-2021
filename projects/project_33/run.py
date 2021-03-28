#!/usr/bin/env python
import os
import sys
import json
import pandas as pd

sys.path.insert(0, 'src/data')
from Loading_Data import * 
sys.path.insert(0, 'src/eda')
from feature_selection import * 
sys.path.insert(0, 'src/model')
from hypothesis_testing import * 



def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'.

    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        # make the data target
        device_use = load_device(data_cfg["DEVICE_OUTFP1"], data_cfg["DEVICE_OUTFP2"])
        battery_event = load_battery_event(data_cfg["BATTERY_EVENT_OUTFP"])
        battery_info = load_battery_info(data_cfg["BATTERY_INFO_OUTFP"])
        process = load_process(data_cfg["PROCESS_OUTFP1"])
        cpu = load_cpu(data_cfg["HW_OUTFP1"], data_cfg["HW_OUTFP2"])
        
        print(device_use)
        print(battery_event)
        print(battery_info)
        print(process)
        print(cpu)

        
    if 'eda' in targets:
        ## Data Preprocessing Part
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        try:
            device_use
        except:
            device_use = load_device(data_cfg["DEVICE_OUTFP1"], data_cfg["DEVICE_OUTFP2"])
        try:
            battery_event
        except:
            battery_event = load_battery_event(data_cfg["BATTERY_EVENT_OUTFP"])
        try:
            battery_info 
        except:
            battery_info = load_battery_info(data_cfg["BATTERY_INFO_OUTFP"])
        try:
            process
        except:
            process = load_process(data_cfg["PROCESS_OUTFP1"])
        try:
            cpu
        except:
            cpu = load_cpu(data_cfg["HW_OUTFP1"], data_cfg["HW_OUTFP2"])
        
        num_dev = num_dev_feature(battery_event, device_use)
        num_proc = num_proc_feature(battery_event, process)
        page_faults = page_faults_feature(battery_event, process)
        avg_memory = avg_memory_feature(battery_event, process)
        cpu_sec = cpu_sec_feature(battery_event, process)
        capacity = capacity_feature(battery_event, battery_info)
        cpu_percent = cpu_percent_feature(cpu, battery_event)
        cpu_temp = cpu_temp_feature(cpu, battery_event)

    
    if 'model' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        try:
            device_use
        except:
            device_use = load_device(data_cfg["DEVICE_OUTFP1"], data_cfg["DEVICE_OUTFP2"])
        try:
            battery_event
        except:
            battery_event = load_battery_event(data_cfg["BATTERY_EVENT_OUTFP"])
        try:
            battery_info 
        except:
            battery_info = load_battery_info(data_cfg["BATTERY_INFO_OUTFP"])
        try:
            process
        except:
            process = load_process(data_cfg["PROCESS_OUTFP1"])
        try:
            cpu
        except:
            cpu = load_cpu(data_cfg["HW_OUTFP1"], data_cfg["HW_OUTFP2"])
            
        try:
            num_dev
        except:
            num_dev = num_dev_feature(battery_event, device_use)
        try:
            num_proc
        except:
            num_proc = num_proc_feature(battery_event, process)
        try:
            page_faults 
        except:
            page_faults = page_faults_feature(battery_event, process)
        try:
            avg_memory
        except:
            avg_memory = avg_memory_feature(battery_event, process)
        try:
            cpu_sec
        except:
            cpu_sec = cpu_sec_feature(battery_event, process)
        try:
            capacity 
        except:
            capacity = capacity_feature(battery_event, battery_info)
        try:
            cpu_percent
        except:
            cpu_percent = cpu_percent_feature(cpu, battery_event)
        try:
            cpu_temp
        except:
            cpu_temp = cpu_temp_feature(cpu, battery_event)
            
        X = pd.concat([num_proc, page_faults, capacity, cpu_percent, cpu_temp, num_dev,avg_memory,cpu_sec], axis = 1).dropna()
        y = battery_event[['guid', 'battery_minutes_remaining']][battery_event.guid.isin(X.index)].groupby('guid')['battery_minutes_remaining'].apply(lambda x: (x!=-1).mean())

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3)

        linear_train, linear_test = linear_reg(X_train1, y_train1, X_test1,  y_test1)
        svm_train, svm_test = supportvm(X_train1, y_train1,X_test1, y_test1)
        dt_train, dt_test = dtr(X_train1, y_train1,X_test1, y_test1)
        rf_train, rf_test = rf( X_train1, y_train1, X_test1, y_test1)
        ada_train, ada_test = ada( X_train1, y_train1, X_test1, y_test1)
        gradient_train, gradient_test = gradient( X_train1, y_train1, X_test1, y_test1)
        
        print('\n')
        hypo1(X,y,gradient_test,svm_test)
        print('\n')
        hypo2(X,y,gradient_test,ada_test)
           
    
    if 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        # make the data target

        device_use = load_device(data_cfg["DEVICE_OUTFP1"], data_cfg["DEVICE_OUTFP2"])
        battery_event = load_battery_event(data_cfg["BATTERY_EVENT_OUTFP"])
        battery_info = load_battery_info(data_cfg["BATTERY_INFO_OUTFP"])
        process = load_process(data_cfg["PROCESS_OUTFP1"])
        cpu = load_cpu(data_cfg["HW_OUTFP1"], data_cfg["HW_OUTFP2"])
        
        print(device_use)
        print(battery_event)
        print(battery_info)
        print(process)
        print(cpu)
        
        num_dev = num_dev_feature(battery_event, device_use)
        num_proc = num_proc_feature(battery_event, process)
        page_faults = page_faults_feature(battery_event, process)
        avg_memory = avg_memory_feature(battery_event, process)
        cpu_sec = cpu_sec_feature(battery_event, process)
        capacity = capacity_feature(battery_event, battery_info)
        cpu_percent = cpu_percent_feature(cpu, battery_event)
        cpu_temp = cpu_temp_feature(cpu, battery_event)

        X = pd.concat([num_proc, page_faults, capacity, cpu_percent, cpu_temp, num_dev,avg_memory,cpu_sec], axis = 1).dropna()
        y = battery_event[['guid', 'battery_minutes_remaining']][battery_event.guid.isin(X.index)].groupby('guid')['battery_minutes_remaining'].apply(lambda x: (x!=-1).mean())

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3)

        linear_train, linear_test = linear_reg(X_train1, y_train1, X_test1,  y_test1)
        svm_train, svm_test = supportvm(X_train1, y_train1,X_test1, y_test1)
        dt_train, dt_test = dtr(X_train1, y_train1,X_test1, y_test1)
        rf_train, rf_test = mae(RandomForestRegressor(), X_train1, y_train1, X_test1, y_test1)
        ada_train, ada_test = mae(AdaBoostRegressor(), X_train1, y_train1, X_test1, y_test1)
        gradient_train, gradient_test = mae(GradientBoostingRegressor(), X_train1, y_train1, X_test1, y_test1)
        
        print('\n')
        hypo1(X,y,gradient_test,svm_test)
        print('\n')
        hypo2(X,y,gradient_test,ada_test)


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)

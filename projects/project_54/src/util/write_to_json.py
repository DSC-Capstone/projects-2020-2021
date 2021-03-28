import json
from datetime import datetime
import os

def write_model_to_json(loss_train, acc_train, fs_train, loss_val, acc_val, fs_val, n_epochs, model_name,fp):
    '''
    write model performance to a json file named
    DATE_EPOCHS_ MODELNAME_performance.json
    '''
    data = {}
    
    # train performance, take last entry as final epoch.
    data['loss_train'] = loss_train[-1]
    data['acc_train'] = acc_train[-1]
    data['fs_train'] = fs_train[-1]
    
    # validation performance
    data['loss_val'] = loss_val[-1]
    data['acc_val'] = acc_val[-1]
    data['fs_val'] = fs_val[-1]
    
    # date
    now = datetime.now()
    now_formatted = now.strftime("%d%m%Y_%H:%M")
    
    # format filename
    file_name = "{}_{}_{}_performance.json".format(now_formatted, n_epochs, model_name)
    write_fp = os.path.join(fp, file_name)
    
    # write to file
    with open(write_fp, 'w') as json_file:
        json.dump(data, json_file)
    
    print('Wrote model performance to {}'.format(write_fp))
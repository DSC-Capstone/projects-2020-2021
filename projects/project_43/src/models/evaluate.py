# Copyright (c) 2017 NVIDIA Corporation
import torch
import copy
from src.encoder.data import input_layer
from src.encoder.model import model
from torch.autograd import Variable
from pathlib import Path
import sys


use_gpu = torch.cuda.is_available() # global flag
if use_gpu:
    print('GPU is available.') 
else: 
    print('GPU is not available.')

def main(configs):
    params = dict()
    params['batch_size'] = 1
    params['data_dir'] =  configs['train_data']
    params['major'] = 'users'
    params['itemIdInd'] = 1
    params['userIdInd'] = 0
    print("Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    print("Loading test data")
    test_params = copy.deepcopy(params)
    # must set test batch size to 1 to make sure no examples are missed
    test_params['batch_size'] = 1
    test_params['data_dir'] = configs['test_data']
    test_data_layer = input_layer.UserItemRecDataProvider(params=test_params,
                                                          user_id_map=data_layer.userIdMap,
                                                          item_id_map=data_layer.itemIdMap)

    rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] + [int(l) for l in configs['HIDDEN'].split(',')],
                                 nl_type=configs['ACTIVATION'],
                                 is_constrained=configs['CONSTRAINED'],
                                 dp_drop_prob=configs['DROPOUT'],
                                 last_layer_activations=not configs['SKIP_LAST_LAYER_NL'])
    
    path_to_model = Path(configs['model_save'])
    if path_to_model.is_file():
        print("Loading model from: {}".format(path_to_model))
        rencoder.load_state_dict(torch.load(configs['model_save']))

    print('######################################################')
    print('######################################################')
    print('############# AutoEncoder Model: #####################')
    print(rencoder)
    print('######################################################')
    print('######################################################')
    rencoder.eval()
    if use_gpu: rencoder = rencoder.cuda()
  
    inv_userIdMap = {v: k for k, v in data_layer.userIdMap.items()}
    inv_itemIdMap = {v: k for k, v in data_layer.itemIdMap.items()}

    test_data_layer.src_data = data_layer.data
    with open(configs['prediction_location'], 'w') as outf:
        for i, ((out, src), majorInd) in enumerate(test_data_layer.iterate_one_epoch_eval(for_inf=True)):
            inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
            targets_np = out.to_dense().numpy()[0, :]
            outputs = rencoder(inputs).cpu().data.numpy()[0, :]
            non_zeros = targets_np.nonzero()[0].tolist()
            major_key = inv_userIdMap [majorInd]
            for ind in non_zeros:
                outf.write("{}\t{}\t{}\t{}\n".format(major_key, inv_itemIdMap[ind], outputs[ind], targets_np[ind]))
            if i % 10000 == 0:
                print("Done: {}".format(i))

if __name__ == '__main__':
    main(sys.argv)


# Copyright (c) 2017 NVIDIA Corporation
import torch
from src.encoder.data import input_layer
from src.encoder.model import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import time
from pathlib import Path
from src.encoder.logger import Logger
from math import sqrt
import numpy as np
import os
import sys

use_gpu = torch.cuda.is_available() # global flag
if use_gpu:
    print('GPU is available.') 
else: 
    print('GPU is not available.')

def do_eval(encoder, validation_data_layer):
    encoder.eval()
    denom = 0.0
    total_epoch_loss = 0.0
    for i, (eval, src) in enumerate(validation_data_layer.iterate_one_epoch_eval()):
        inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
        targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
        outputs = encoder(inputs)
        loss, num_ratings = model.MSEloss(outputs, targets)
        total_epoch_loss += loss.item()
        denom += num_ratings.item()
    return sqrt(total_epoch_loss / denom)

def log_var_and_grad_summaries(logger, layers, global_step, prefix, log_histograms=False):
    """
    Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
    :param logger: TB logger
    :param layers: param list
    :param global_step: global step for TB
    :param prefix: name prefix
    :param log_histograms: (default: False) whether or not log histograms
    :return:
    """
    for ind, w in enumerate(layers):
    # Variables
        w_var = w.data.cpu().numpy()
        logger.scalar_summary("Variables/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_var),
                              global_step)
        if log_histograms:
            logger.histo_summary(tag="Variables/{}_{}".format(prefix, ind), values=w.data.cpu().numpy(),
                               step=global_step)

    # Gradients
        w_grad = w.grad.data.cpu().numpy()
        logger.scalar_summary("Gradients/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_grad),
                              global_step)
        if log_histograms:
            logger.histo_summary(tag="Gradients/{}_{}".format(prefix, ind), values=w.grad.data.cpu().numpy(),
                             step=global_step)

def main(configs):
    logger = Logger(configs['model_output_dir'])
    params = dict()
    params['batch_size'] = configs['BATCH_SIZE']
    params['data_dir'] =  configs['train_data']
    params['major'] = 'users'
    params['itemIdInd'] = 1
    params['userIdInd'] = 0
    print("Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    print("Loading valid data")
    valid_params = copy.deepcopy(params)
    # must set valid batch size to 1 to make sure no examples are missed
    valid_params['data_dir'] = configs['valid_data']
    valid_data_layer = input_layer.UserItemRecDataProvider(params=valid_params,
                                                        user_id_map=data_layer.userIdMap, # the mappings are provided
                                                        item_id_map=data_layer.itemIdMap)
    valid_data_layer.src_data = data_layer.data
    rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] + [int(l) for l in configs['HIDDEN'].split(',')],
                               nl_type=configs['ACTIVATION'],
                               is_constrained=configs['CONSTRAINED'],
                               dp_drop_prob=configs['DROPOUT'],
                               last_layer_activations=not configs['SKIP_LAST_LAYER_NL'])
    model_checkpoint = configs['model_output_dir'] + "/model"
    path_to_model = Path(model_checkpoint)
    if path_to_model.is_file():
        print("Loading model from: {}".format(model_checkpoint))
        rencoder.load_state_dict(torch.load(model_checkpoint))

    print('######################################################')
    print('######################################################')
    print('############# AutoEncoder Model: #####################')
    print(rencoder)
    print('######################################################')
    print('######################################################')

    gpu_ids = [int(g) for g in configs['GPUS'].split(',')]
    print('Using GPUs: {}'.format(gpu_ids))
    if len(gpu_ids)>1:
        rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)
  
    if use_gpu: rencoder = rencoder.cuda()

    if configs['OPTIMIZER'] == "adam":
        optimizer = optim.Adam(rencoder.parameters(),
                               lr=configs['LR'],
                               weight_decay=configs['WD'])
    elif configs['OPTIMIZER'] == "adagrad":
        optimizer = optim.Adagrad(rencoder.parameters(),
                                  lr=configs['LR'],
                                  weight_decay=configs['WD'])
    elif configs['OPTIMIZER'] == "momentum":
        optimizer = optim.SGD(rencoder.parameters(),
                              lr=configs['LR'], momentum=0.9,
                              weight_decay=configs['WD'])
        scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
    elif configs['OPTIMIZER'] == "rmsprop":
        optimizer = optim.RMSprop(rencoder.parameters(),
                                  lr=configs['LR'], momentum=0.9,
                                  weight_decay=configs['WD'])
    else:
        raise  ValueError('Unknown optimizer kind')

    t_loss = 0.0
    t_loss_denom = 0.0
    global_step = 0

    if configs['NOISE_PROB'] > 0.0:
        dp = nn.Dropout(p=configs['NOISE_PROB'])

    for epoch in range(configs['EPOCHS']):
        print('Doing epoch {} of {}'.format(epoch, configs['EPOCHS']))
        e_start_time = time.time()
        rencoder.train()
        total_epoch_loss = 0.0
        denom = 0.0
        if configs['OPTIMIZER'] == "momentum":
            scheduler.step()
        for i, mb in enumerate(data_layer.iterate_one_epoch()):
            inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
            optimizer.zero_grad()
            outputs = rencoder(inputs)
            loss, num_ratings = model.MSEloss(outputs, inputs)
            loss = loss / num_ratings
            loss.backward()
            optimizer.step()
            global_step += 1
            t_loss += loss.item()
            t_loss_denom += 1

            if i % configs['SUMMARY_FREQUENCY'] == 0:
                print('[%d, %5d] RMSE: %.7f' % (epoch, i, sqrt(t_loss / t_loss_denom)))
                logger.scalar_summary("Training_RMSE", sqrt(t_loss/t_loss_denom), global_step)
                t_loss = 0
                t_loss_denom = 0.0
                log_var_and_grad_summaries(logger, rencoder.encode_w, global_step, "Encode_W")
                log_var_and_grad_summaries(logger, rencoder.encode_b, global_step, "Encode_b")
                if not rencoder.is_constrained:
                    log_var_and_grad_summaries(logger, rencoder.decode_w, global_step, "Decode_W")
                log_var_and_grad_summaries(logger, rencoder.decode_b, global_step, "Decode_b")

            total_epoch_loss += loss.item()
            denom += 1

          #if configs['AUG_STEP'] > 0 and i % configs['AUG_STEP'] == 0 and i > 0:
            if configs['AUG_STEP'] > 0:
            # Magic data augmentation trick happen here
                for t in range(configs['AUG_STEP']):
                    inputs = Variable(outputs.data)
                    if configs['NOISE_PROB'] > 0.0:
                        inputs = dp(inputs)
                    optimizer.zero_grad()
                    outputs = rencoder(inputs)
                    loss, num_ratings = model.MSEloss(outputs, inputs)
                    loss = loss / num_ratings
                    loss.backward()
                    optimizer.step()

        e_end_time = time.time()
        print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
              .format(epoch, e_end_time - e_start_time, sqrt(total_epoch_loss/denom)))
        logger.scalar_summary("Training_RMSE_per_epoch", sqrt(total_epoch_loss/denom), epoch)
        logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
        print("Saving model to {}".format(model_checkpoint + ".epoch_"+str(epoch)))
        torch.save(rencoder.state_dict(), model_checkpoint + ".epoch_"+str(epoch))

    print("Saving model to {}".format(model_checkpoint + ".last"))
    torch.save(rencoder.state_dict(), model_checkpoint + ".last")

    # save to onnx
    dummy_input = Variable(torch.randn(params['batch_size'], data_layer.vector_dim).type(torch.float))
    torch.onnx.export(rencoder.float(), dummy_input.cuda() if use_gpu else dummy_input, 
                    model_checkpoint + ".onnx", verbose=True)
    print("ONNX model saved to {}!".format(model_checkpoint + ".onnx"))

if __name__ == '__main__':
    main(sys.argv)
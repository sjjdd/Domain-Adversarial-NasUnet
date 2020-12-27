# -*- coding: utf-8 -*-


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
from dann_helper import DANN
from build_model import train_SegmentationModel  #pylint:disable = E0401
from data_loder_samples import SamplesDataLoaderNAS  #pylint:disable = E0401

# from utils.keras_config import set_keras_config, get_available_gpus_num  #pylint:disable = E0401,E0611
from lr_tricks import LearningRateFinder, CyclicCosineRestart  #pylint:disable = E0401,E0611
from grl_tricks import GRL_RATE
from dann_helper import DANN

path_to_train_image_s = r'C:\Users\84471\Desktop\test\source\img'
path_to_train_image_t = r'C:\Users\84471\Desktop\test\target'
path_to_train_labels = r'C:\Users\84471\Desktop\test\source\label'

path_to_valid_image_s = r'C:\Users\84471\Desktop\test\val\img'
path_to_valid_image_t = r'C:\Users\84471\Desktop\test\val\img'
path_to_valid_labels = r'C:\Users\84471\Desktop\test\val\label'

pretrain_weight = r'E:\learning\pythonproject\world_model\trained_model\single_model\stage_0\check_point\weights_191.hdf5'
# pretrain_weight = ''

input_size = 256
optimizer_lr = 3e-6
# optimizer_lr = 0
batchsize = 1    #(s,t)
num_epoches = 300

# lr_callback = CyclicCosineRestart(lr_min=1e-5,
#                                     lr_max=optimizer_lr,
#                                     number_of_lr_warm_epochs=10,
#                                     number_of_epochs=100,
#                                     use_warmup=False)

grl_callback = GRL_RATE(number_of_epochs = num_epoches)
callback = [grl_callback]
train_SegmentationModel(
    DANN(summary=True, width=input_size, height=input_size, channels=3, classes=2, batch_size=2*batchsize, model_plot=False,grl = 1.0).model,
    SamplesDataLoaderNAS,
    path_to_train_image_s,
    path_to_train_image_t,
    path_to_train_labels,
    path_to_valid_image_s,
    path_to_valid_image_t,
    path_to_valid_labels,
    num_gpu=1,
    workers=2,
    batch_size=batchsize,
    learning_rate=optimizer_lr,
    checkpoint_dir='check_point',
    pretrain_weights=pretrain_weight,
    num_epoches=num_epoches,
    train_input_size=(input_size,input_size),
    train_input_stride=(input_size,input_size),
    valid_input_size=(input_size,input_size),
    valid_input_stride=(input_size,input_size),
    num_classes=2,
    input_bands=3,
    class_weights=None,
    loss_weights=None,
    custom_callback=callback,
    custom_loss=None,
    period=num_epoches

)

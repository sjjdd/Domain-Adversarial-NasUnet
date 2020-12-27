# -*- coding: utf-8 -*-

import numpy as np
from skimage.io import imsave, imread

from keras.models import Model
import os
import matplotlib.pyplot as plt
from keras.applications.nasnet import preprocess_input

from net_architecture.NASUnetModel import NAS_U_Net
from NASNet_prediction_view import NASUnet_predict
import fnmatch
from dann_helper import DANN

imgsize = 256
batch_size = 1
seg_model = DANN(summary=True, width=imgsize, height=imgsize, channels=3, classes=2, batch_size=batch_size, model_plot=False).model
# path_to_valid_image = r'F:\Data\ZT_SRDataset\四个定量精度评价区\IMG2p5'
# path_to_valid_labels = r'F:\Data\ZT_SRDataset\四个定量精度评价区\GT2p5TF'
path_to_valid_image = r'C:\Users\84471\Desktop\test\source\img'
# path_to_valid_labels = r'C:\Users\84471\Desktop\test\source\label'
outpath = r'C:\Users\84471\Desktop\test\adver2'
pretrained_weights = r'E:\xview2\adversarial_training\Keras-NASNet-master\check_point\weights_300.hdf5'


model =NASUnet_predict(seg_model,pretrained_weights,2,preprocess_input=preprocess_input,input_size=256,startX=0,startY=0,
                       upscale=1)
file_names = [
    file_name for file_name in os.listdir(path_to_valid_image)
    if fnmatch.fnmatch(file_name, '*.png') and 'cloud' not in file_name
]
for i, file_name in enumerate(file_names):
    model.predict_and_save_prediction(
        input_folder=path_to_valid_image,
        labels_folder=None,
        output_folder=outpath,
        file_name=file_name,
        label_format='png',
        upsample=1
    )

    print(str(i)+'th image is done..')
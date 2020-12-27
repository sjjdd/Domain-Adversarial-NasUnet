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
from data_loder_samples import SamplesDataLoaderNAS


path_to_valid_image_s = r'C:\Users\84471\Desktop\test\source\img'
path_to_valid_image_t = r'C:\Users\84471\Desktop\test\target'
path_to_valid_label_s = r'C:\Users\84471\Desktop\test\source\label'
outpath = r'C:\Users\84471\Desktop\test\result_test'
pretrained_weights = r'E:\xview2\adversarial_training\Keras-NASNet-master\check_point\weights_300.hdf5'

imgsize = 256
batch_size = 1


model = DANN(summary=True, width=imgsize, height=imgsize, channels=3, classes=2, batch_size=batch_size, model_plot=False).model
model.load_weights(pretrained_weights, by_name=True)


test_generator_params = {
    'xs_set_dir': path_to_valid_image_s,
    'xt_set_dir': path_to_valid_image_t,
    'y_set_dir' : path_to_valid_label_s,
    'patch_size': (imgsize,imgsize),
    'patch_stride': (imgsize,imgsize),
    'batch_size': batch_size
}
test_generator = SamplesDataLoaderNAS(**test_generator_params)
test_generator.reset()
a = model.predict_generator(test_generator)
print(a)
a = a[1]
for i in range(2):
    img_result = a[i].reshape(imgsize, imgsize,2)
    # convert probability to image
    img_max = np.argmax(img_result, axis=-1)
    tile_label = np.zeros(
        (imgsize, imgsize), dtype=np.uint8)
    tile_label[img_max == 1] = 255
    imsave(os.path.join(outpath, str(i) + '.tif'), tile_label)
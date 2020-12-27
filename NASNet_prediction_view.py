# -*- coding: utf-8 -*-
"""
@ Time    : 2020/7/1 7:52
@ Author  : Xu Penglei
@ Email   : xupenglei87@163.com
@ File    : NASNet_prediction_view.py
@ Desc    : None
"""
import numpy as np
from skimage.io import imsave, imread

from keras.models import Model
from keras.layers import Input
import os
import matplotlib.pyplot as plt
from keras.applications.nasnet import preprocess_input
import cv2


class NASUnet_predict():
    def __init__(self,
                 segmentation_model,
                 pretrain_weight,
                 number_of_class,
                 input_size,
                 preprocess_input=preprocess_input,
                 num_bands=3,
                 startX=0,
                 startY=0,
                 upscale=1
                 ):
        self.segmentation_model = segmentation_model
        self.pretrain_weight = pretrain_weight
        self.number_of_class = number_of_class
        self.preprocess_input = preprocess_input
        self.input_size = input_size
        self.num_bands = num_bands
        self.startX = startX
        self.startY = startY
        self.upscale = upscale
        self.load_model_geoboost_model()

    def load_model_geoboost_model(self
                                  ):
        img_input = Input((self.input_size*self.upscale,self.input_size*self.upscale,self.num_bands))
        # logits, out = self.segmentation_model(img_input, self.number_of_class)
        # out = self.segmentation_model(img_input, self.number_of_class)
        # model = Model(img_input, out)
        model = self.segmentation_model
        print('Load weights from %s' % (self.pretrain_weight))
        model.load_weights(self.pretrain_weight, by_name=True)
        self.model = model

    def get_input(self, file_name, input_folder,SR=1):
        image_path = os.path.join(input_folder, file_name)
        # img = imread(image_path).astype('float32').repeat(2, axis=0).repeat(2, axis=1)
        img_t = imread(image_path)[self.startX:self.startX+self.input_size,self.startY:self.startY+self.input_size]
        if SR>1:
            img_t = cv2.resize(img_t,(0,0),fx=SR,fy=SR)
        img = self.preprocess_input(img_t.astype('float32'))

        img_list = np.array([img])

        return img_list,img_t

    def get_label(self, file_name, input_folder):
        image_path = os.path.join(input_folder, file_name)
        img = imread(image_path).astype('float32')[self.startX:self.startX + self.input_size,
              self.startY:self.startY + self.input_size]
        return img

    def predict_and_save_prediction(self, input_folder, labels_folder, output_folder, file_name, label_format='tif',upsample=1):

        # input preprocessing
        input_list,img_t = self.get_input(file_name, input_folder,upsample)
        if labels_folder:
            if 'cloud' in file_name:
                label = self.get_label(file_name.replace('.tif','_HR.'+label_format),labels_folder)
            else:
                label = self.get_label(file_name.replace('tif', label_format), labels_folder)

        # predict
        result = self.model.predict(input_list, batch_size=1)
        img_prob = result[1]
        img_result = img_prob.reshape(self.input_size*self.upscale, self.input_size*self.upscale,
                                      self.number_of_class)

        # convert probability to image
        img_max = np.argmax(img_result, axis=-1)
        tile_label = np.zeros(
            (self.input_size*self.upscale, self.input_size*self.upscale), dtype=np.uint8)
        tile_label[img_max == 1] = 255

        if output_folder:
            imsave(os.path.join(output_folder, file_name[:-3]+'tif'), tile_label)
        else:
            fig,ax = plt.subplots(1,3,figsize=(12,8))
            ax[0].imshow(tile_label,cmap='gray')
            ax[0].set_title('pred')
            ax[1].imshow(img_t.astype('uint8'))
            ax[1].set_title('IMG')
            ax[2].imshow(label,cmap='gray')
            ax[2].set_title('gt')
            plt.show()

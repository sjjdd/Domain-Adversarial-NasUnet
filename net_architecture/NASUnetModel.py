# -*- coding: utf-8 -*-
"""
@ Time    : 2020/6/30 20:58
@ Author  :
@ Email   :
@ File    : NASUnetMode.py
@ Desc    : NAS 为Backbone的Unet模型架构
"""
#import keras.backend as K
from keras import backend as K
from keras import layers
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Lambda, Reshape, SeparableConv2D, SpatialDropout2D,
                          UpSampling2D,Input)
from keras.models import Model
from keras.regularizers import l2

from net_architecture.nasnet_keras_SE_nodropout import NASNetMobile, _normal_a_cell, NASNetLarge
from net_architecture.SE_block import csSE_block

bn_momentum = 0.9997
interpolation = 'bilinear'
num_blocks = 4

def transition_up(x,
                  p,
                  num_filter,
                  block_id,
                  weight_decay=0.0):
    x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, num_filter, block_id=f'{block_id}_{i}')
    x = csSE_block(x, block_id)
    return x


def NAS_U_Net(img_input, number_of_class, weight_decay=0.0):

    # img_input = Input(inputshape)
    # base_model = NASNetLarge(include_top=False,
    #                           weights='imagenet',
    #                           input_tensor=img_input)
    base_model = NASNetMobile(include_top=False,
                              weights='imagenet',
                              input_tensor=img_input)

    normal_cell_4_x = base_model.get_layer('res_4x').output
    normal_cell_8_x = base_model.get_layer('res_8x').output

    normal_cell_16_x = base_model.get_layer('res_16x').output

    normal_cell_32_x = base_model.get_layer('res_32x').output

    penultimate_filters = 1056
    filters = penultimate_filters // 24
    filter_multiplier = 2

    x = transition_up(normal_cell_32_x,
                      normal_cell_16_x,
                      filters * filter_multiplier,
                      '32_to_16',
                      weight_decay=weight_decay)

    x = transition_up(x,
                      normal_cell_8_x,
                      filters,
                      '16_to_8',
                      weight_decay=weight_decay)


    x = transition_up(x,
                      normal_cell_4_x,
                      filters // filter_multiplier,
                      '8_to_4',
                      weight_decay=weight_decay)

    x = Activation('relu')(x)

    #x = SpatialDropout2D(0.05)(x)
    x = Conv2D(filters=number_of_class,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=True,
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name='last_conv')(x)
    x = UpSampling2D(size=(4, 4), interpolation=interpolation)(x)
    a = Reshape((-1, number_of_class))(x)
    out = Activation('softmax',name='out')(a)

    return x,a,out

def NAS_large_U_Net(img_input, number_of_class, weight_decay=0.0):

    # img_input = Input(inputshape)
    # base_model = NASNetLarge(include_top=False,
    #                           weights='imagenet',
    #                           input_tensor=img_input)
    base_model = NASNetLarge(include_top=False,
                              weights='imagenet',
                              input_tensor=img_input)

    normal_cell_4_x = base_model.get_layer('res_4x').output
    normal_cell_8_x = base_model.get_layer('res_8x').output

    normal_cell_16_x = base_model.get_layer('res_16x').output

    normal_cell_32_x = base_model.get_layer('res_32x').output

    penultimate_filters = 1056
    filters = penultimate_filters // 24
    filter_multiplier = 2

    x = transition_up(normal_cell_32_x,
                      normal_cell_16_x,
                      filters * filter_multiplier,
                      '32_to_16',
                      weight_decay=weight_decay)

    x = transition_up(x,
                      normal_cell_8_x,
                      filters,
                      '16_to_8',
                      weight_decay=weight_decay)


    x = transition_up(x,
                      normal_cell_4_x,
                      filters // filter_multiplier,
                      '8_to_4',
                      weight_decay=weight_decay)

    x = Activation('relu')(x)

    #x = SpatialDropout2D(0.05)(x)
    x = Conv2D(filters=number_of_class,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=True,
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name='last_conv')(x)
    x = UpSampling2D(size=(4, 4), interpolation=interpolation)(x)
    x = Reshape((-1, number_of_class))(x)
    out = Activation('softmax',name='out')(x)

    return x, out

# -*- coding: utf-8 -*-
"""
@ Time    : 2020/6/30 21:18
@ Author  :
@ Email   :
@ File    : BuildAndTrainSegModel.py
@ Desc    : 构建语义分割模型，同时
"""
from keras import backend as K
from keras import layers, models
import os
import gc
import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, TensorBoard
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model
from skimage.io import imsave, imread
# from metrics import m_iou,m_iou_0,m_iou_1
from multi_gpu import MultiGPUModelCheckpoint
from data_loder_samples import SamplesDataLoaderNAS


# def build_SegmentationModel(seg_model,input_shape,num_classes,pretrain_weights,by_name=True):
#     img_input = layers.Input(input_shape)
#     logits,out = seg_model(img_input,num_classes)
#     model = models.Model(img_input,out)
#     if os.path.exists(pretrain_weights):
#         print('Load weights from %s'%(pretrain_weights))
#         model.load_weights(pretrain_weights,by_name=by_name)
#     return model

def train_SegmentationModel(seg_model,DataLoader,
                            path_to_train_image_s,
                            path_to_train_image_t,
                            path_to_train_labels,
                            path_to_valid_image_s,
                            path_to_valid_image_t,
                            path_to_valid_labels,
                            num_gpu=0,
                            workers=2,
                            batch_size=1,
                            learning_rate=3e-4,
                            checkpoint_dir='.check_point',
                            pretrain_weights= '',
                            num_epoches=100,
                            train_input_size=(256,256),
                            train_input_stride=(256,256),
                            valid_input_size=(256,256),
                            valid_input_stride=(256,256),
                            num_classes=2,
                            input_bands=3,
                            class_weights=None,
                            loss_weights=None,
                            custom_callback=None,
                            custom_loss=None,
                            period = 1

                            ):
    train_generator_params = {
        'xs_set_dir': path_to_train_image_s,
        'xt_set_dir': path_to_train_image_t,
        'y_set_dir': path_to_train_labels,
        'patch_size': train_input_size,
        'patch_stride': train_input_stride,
        'batch_size': batch_size,
        'shuffle': True,

        'is_train': True
    }
    test_generator_params = {
        'xs_set_dir': path_to_valid_image_s,
        'xt_set_dir': path_to_valid_image_t,
        'y_set_dir': path_to_valid_labels,
        'patch_size': valid_input_size,
        'patch_stride': valid_input_stride,
        'batch_size': batch_size
    }

    if os.path.exists(pretrain_weights):
        print('Load weights from %s'%(pretrain_weights))
        seg_model.load_weights(pretrain_weights,by_name=True)

    if num_gpu == 1:
        # model = build_SegmentationModel(
        #     seg_model,
        #     (*train_input_size,input_bands),
        #     num_classes,
        #     pretrain_weights
        # )
        model = seg_model
        print("Training using one GPU..")
    else:
        with tf.device('/cpu:0'):
            # model = build_SegmentationModel(
            #     seg_model,
            #     (*train_input_size, input_bands),
            #     num_classes,
            #     pretrain_weights
            # )
            model = seg_model
    if num_gpu>1:
        # parallel_model = multi_gpu_model(model, gpus=num_gpu)\
        parallel_model = model
        print("Training using multiple GPUs..")
    else:
        parallel_model = model
        print("Training using one GPU or CPU..")

    # for layer in parallel_model.layers:
    #     layer.trainable = False

    parallel_model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate),
                        #optimizer=keras.optimizers.SGD(learning_rate),
                       loss={'classifier_output': 'categorical_crossentropy', 'discriminator_output': 'binary_crossentropy'},
                       # loss={'classifier_output': 'categorical_crossentropy'}
                       loss_weights={'classifier_output': 5.0, 'discriminator_output': 1.0}
                           )


    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensor_boarder = TensorBoard(log_dir=checkpoint_dir, update_freq='epoch')
    csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training.log'))
    checkpointer = MultiGPUModelCheckpoint(cpu_model=model,
                                           filepath=os.path.join(
                                               checkpoint_dir, 'weights_{epoch:03d}.hdf5'),
                                           verbose=1,
                                           monitor='val_acc',
                                           mode='max',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           weights_path_list=[''],
                                           period=period
                                           )

    call_back_list = [tensor_boarder, checkpointer, csv_logger]
    if not (custom_callback is None):
        call_back_list.extend(custom_callback)


    train_generator = DataLoader(**train_generator_params)
    # test_generator = DataLoader(**test_generator_params)

    parallel_model.fit_generator(train_generator,
                                 epochs=num_epoches,
                                 workers=workers,
                                 # workers=1,
                                 verbose=1,
                                 use_multiprocessing=False,
                                 # validation_data=test_generator,
                                 max_queue_size=workers,
                                 callbacks=call_back_list,
                                 class_weight=class_weights)






    path_to_valid_image_s = r'C:\Users\84471\Desktop\test\source\img'
    path_to_valid_image_t = r'C:\Users\84471\Desktop\test\target'
    path_to_valid_label_s = r'C:\Users\84471\Desktop\test\source\label'
    outpath = r'C:\Users\84471\Desktop\test\result1'
    pretrained_weights = r'E:\xview2\adversarial_training\Keras-NASNet-master\check_point\weights_300.hdf5'

    imgsize = 256
    batch_size = 1

    test_generator_params = {
        'xs_set_dir': path_to_valid_image_s,
        'xt_set_dir': path_to_valid_image_t,
        'y_set_dir': path_to_valid_label_s,
        'patch_size': (imgsize, imgsize),
        'patch_stride': (imgsize, imgsize),
        'batch_size': batch_size
    }
    test_generator = SamplesDataLoaderNAS(**test_generator_params)
    test_generator.reset()
    a = parallel_model.predict_generator(test_generator)
    print(a)
    a = a[1]
    for i in range(2):
        img_result = a[i].reshape(imgsize, imgsize, 2)
        # convert probability to image
        img_max = np.argmax(img_result, axis=-1)
        tile_label = np.zeros(
            (imgsize, imgsize), dtype=np.uint8)
        tile_label[img_max == 1] = 255
        imsave(os.path.join(outpath, str(i) + '.tif'), tile_label)
    K.clear_session()

# def evaluate_SegmentationModel(
#         seg_model,DataLoader,
#         path_to_valid_image,
#         path_to_valid_labels,
#         pretrain_weights,
#         file_name=None,
#         test_file_names_list=None,
#         num_gpu=0,
#         workers=2,
#         batch_size=4,
#         learning_rate=3e-4,
#         valid_input_size=(512,512),
#         valid_input_stride=(512,512),
#         num_classes=2,
#         input_bands=3,
#         class_weights=None,
#         loss_weights=None,
#         custom_callback=None,
#         custom_loss=None
#                             ):
#     test_generator_params = {
#         'x_set_dir': path_to_valid_image,
#         'y_set_dir': path_to_valid_labels,
#         'patch_size': valid_input_size,
#         'patch_stride': valid_input_stride,
#         'batch_size': batch_size,
#         'file_names': test_file_names_list
#     }
#     if num_gpu == 1:
#         model = build_SegmentationModel(
#             seg_model,
#             (*valid_input_size,input_bands),
#             num_classes,
#             pretrain_weights
#         )
#         print("Training using one GPU..")
#     else:
#         with tf.device('/cpu:0'):
#             model = build_SegmentationModel(
#                 seg_model,
#                 (*valid_input_size, input_bands),
#                 num_classes,
#                 pretrain_weights
#             )
#     if num_gpu>1:
#         parallel_model = multi_gpu_model(model, gpus=num_gpu)
#         print("Training using multiple GPUs..")
#     else:
#         parallel_model = model
#         print("Training using one GPU or CPU..")
#
#     if custom_loss is None:
#         training_loss = 'categorical_crossentropy'
#     else:
#         training_loss = custom_loss
#
#     parallel_model.compile(loss=training_loss,
#                            optimizer=keras.optimizers.Adam(learning_rate),
#                            metrics=["accuracy", m_iou, m_iou_0, m_iou_1],
#                            loss_weights=loss_weights)
#
#     test_generator = DataLoader(**test_generator_params)
#
#     score = parallel_model.evaluate_generator(test_generator,
#                                               verbose=1,
#                                               workers=workers,
#                                               use_multiprocessing=True)
#     if file_name:
#         with open(file_name,"a") as f:
#             f.writelines(str(score)+'\n')
#     else:
#         print(score)
#     K.clear_session()

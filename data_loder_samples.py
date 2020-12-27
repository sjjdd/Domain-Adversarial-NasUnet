# -*- coding: utf-8 -*-
"""
@ Time    : 2020/6/30 14:49
@ Author  : Xu Penglei
@ Email   : xupenglei87@163.com
@ File    : data_loder_SR.py
@ Desc    : 用于载入样本的Data loader，输入文件夹中的图像不是整幅大图，而是已经分割好的样本，如每幅图是256*256
"""
import fnmatch
import os
import random
import keras

# import imageio
import imgaug as ia
import numpy as np
import skimage.io as io
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from keras_preprocessing.image import random_brightness
from keras.applications.nasnet import preprocess_input
from skimage.io import imread
from skimage.morphology import dilation, disk
import gc
# from modelCode.data_loader.data_augmentation.augmentation import random_channel_shift
# from data_loader.data_augmentation.augmentation import random_channel_shift
from net_architecture import get_preprocessing

class SamplesDataLoader(Sequence):
    def __init__(self,
                 xs_set_dir,
                 xt_set_dir,
                 y_set_dir,
                 patch_size=256,
                 patch_stride=256,
                 batch_size=256,
                 shuffle=False,
                 is_train=False,
                 num_classes=2,
                 is_multi_scale=False,
                 img_format='png',
                 label_format='png',
                 backbone=None):

        self.file_names_s = [
            file_name for file_name in os.listdir(xs_set_dir)
            if fnmatch.fnmatch(file_name, '*.%s'%(img_format))
        ]

        self.file_names_d= [
            file_name for file_name in os.listdir(xt_set_dir)
            if fnmatch.fnmatch(file_name, '*.%s'%(img_format))
        ]

        self.backbone=backbone

        self.images_filenames_s = [
            os.path.join(xs_set_dir, item) for item in self.file_names_s
        ]
        self.images_filenames_t = [
            os.path.join(xt_set_dir, item) for item in self.file_names_d
        ]


        self.labels_filenames = []
        for item in self.file_names_s:
            if 'cloud' in item:
                t = item.split('.')[0]+'_HR'+'.'+label_format
                self.labels_filenames.append(os.path.join(y_set_dir,t))
            else:
                self.labels_filenames.append(os.path.join(y_set_dir, item.replace('tif',label_format)))


        img = imread(self.images_filenames_s[0], plugin='gdal').astype(np.uint8)
        # img = imread(self.images_filenames[0]).astype(np.uint8)
        self.image_height = img.shape[0]
        self.image_width = img.shape[1]
        self.num_bands = img.shape[2]

        self.is_train = is_train
        self.is_multi_scale = is_multi_scale

        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]
        if self.is_train:
            # 2.14是用于数据增强的经验值
            # self.patch_height = int(patch_size[0] * 2.14)
            # self.patch_width = int(patch_size[1] * 2.14)

            self.sample_height = patch_size[0]
            self.sample_width = patch_size[1]
        # else:
        #     self.patch_height = patch_size[0]
        #     self.patch_width = patch_size[1]

        self.patch_height_stride = patch_stride[0]
        self.patch_width_stride = patch_stride[1]

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_classes = num_classes
        # if self.is_train:
        #     self.select_valid_sample()
        # else:
        #     self.patches_index =  np.arange(0, self.num_patches)
        self.patches_index = np.arange(0, self.num_patches)
        self.reset()


    @property
    def patch_rows_per_img(self):
        return int((self.image_height - self.patch_height) /
                   self.patch_height_stride) + 1
    @property
    def patch_cols_per_img(self):
        return int((self.image_width - self.patch_width) /
                   self.patch_width_stride) + 1
    @property
    def patches_per_img(self):
        return self.patch_rows_per_img * self.patch_cols_per_img
    @property
    def num_imgs(self):
        return len(self.images_filenames_s)
    @property
    def num_patches(self):
        return self.patches_per_img * self.num_imgs




    def _get_patch(self, filenames, patch_idx):
        img_idx = int(patch_idx / self.patches_per_img)
        img_patch_idx = patch_idx % self.patches_per_img
        row_idx = int(img_patch_idx / self.patch_cols_per_img)
        col_idx = img_patch_idx % self.patch_cols_per_img
        img = imread(filenames[img_idx], plugin='gdal').astype(np.uint8)
        # img = imread(filenames[img_idx]).astype(np.uint8)
        if len(img.shape) > 2:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                                                                 self.patch_height_stride +
                                                                 self.patch_height, col_idx *
                                                                                    self.patch_width_stride:col_idx *
                                                                                                            self.patch_width_stride +
                                                                                                            self.patch_width,
                          :].copy()
        else:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                                                                 self.patch_height_stride +
                                                                 self.patch_height, col_idx *
                                                                                    self.patch_width_stride:col_idx *
                                                                                                            self.patch_width_stride +
                                                                                                            self.patch_width].copy()
        # del img
        # gc.collect()
        return patch_image

    def get_patch(self, filenames, patch_idx):
        return self._get_patch(filenames, patch_idx)

    def data_augmentation(self, image, label):

        # crop_size = random.randint(int(0.8 * self.sample_height),
        #                            int(1.2 * self.sample_height))
        #
        # start_h = random.randint(0, image.shape[0] - int(1.42 * crop_size) - 2)
        # start_w = random.randint(0, image.shape[1] - int(1.42 * crop_size) - 2)
        #
        # image_crop = image[start_h:start_h +
        #                            int(1.42 * crop_size), start_w:start_w +
        #                                                           int(1.42 * crop_size)].copy()
        # label_crop = label[start_h:start_h +
        #                            int(1.42 * crop_size), start_w:start_w +
        #                                                           int(1.42 * crop_size)].copy()
        image_crop = image.copy()
        label_crop = label.copy()

        # del image
        # del label
        # gc.collect()

        seq = iaa.Sequential([
            iaa.Affine(shear=(-4, 4), rotate=(
                0, 360)),  # rotate by -45 to 45 degrees (affects segmaps)
        ])
        segmap = ia.SegmentationMapOnImage(label_crop,
                                           shape=label_crop.shape,
                                           nb_classes=self.num_classes)

        seq_det = seq.to_deterministic()

        image_rotation = seq_det.augment_image(image_crop)
        segmap_aug = seq_det.augment_segmentation_maps(segmap)

        label_rotation = segmap_aug.get_arr_int()

        reduction_pixels = int(0.15 * label_rotation.shape[0])
        start_i = reduction_pixels
        stop_i = label_crop.shape[0] - reduction_pixels
        image_crop = image_rotation[start_i:stop_i, start_i:stop_i, :]
        label_crop = label_rotation[start_i:stop_i, start_i:stop_i]

        seq = iaa.Sequential([
            iaa.Resize(
                {
                    "height": self.sample_height,
                    "width": self.sample_width
                },
                interpolation='nearest'),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.8, iaa.HistogramEqualization()),
            iaa.Sometimes(
                0.8, iaa.CoarseDropout((0.0, 0.05),
                                       size_percent=(0.02, 0.25))),
            iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        ])
        segmap = ia.SegmentationMapOnImage(label_crop,
                                           shape=label_crop.shape,
                                           nb_classes=self.num_classes)

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_image(image_crop)
        segmap_aug = seq_det.augment_segmentation_maps(segmap)

        label_aug = segmap_aug.get_arr_int()

        return image_aug, label_aug

    def preprocess_input(self, img):
        img = img.astype('float32')
        img -= 128.0
        img /= 128.0
        return img

    def preprocess_label(self, label):
        label = label.reshape((-1,))
        # 255 is buildings in inria. 0 is non-buildings.
        label[label > 0] = 1
        label_one_hot = to_categorical(label, num_classes=self.num_classes)

        label_one_hot = label_one_hot.astype('float32')

        return label_one_hot

    def get_batch_patches(self, batch_index):
        batch_patch_idx = self.patches_index[batch_index *
                                             self.batch_size:(batch_index +
                                                              1) *
                                                             self.batch_size]

        batch_image_s = []
        batch_image_t = []
        batch_label = []
        for patch_idx in batch_patch_idx:
            image_s = self.get_patch(self.images_filenames_s,patch_idx)
            image_t = self.get_patch(self.images_filenames_t, patch_idx)
            label = self.get_patch(self.labels_filenames, patch_idx)
            label[label > 0] = 1
            # 由于样本已经是被分割完毕小图，所以即使使用数据增强，也只能得到比原始图像更小范围内的数据，得不到更大范围的数据
            # if self.is_train:
            #     image, label = self.data_augmentation(image, label)

            image_s = self.preprocess_input(image_s)
            image_t = self.preprocess_input(image_t)
            label = self.preprocess_label(label)
            print(label.shape)

            batch_image_s.append(image_s)
            batch_image_t.append(image_t)
            batch_label.append(label)

        batch_image_s = np.array(batch_image_s)
        batch_image_t = np.array(batch_image_t)
        batch_label = np.array(batch_label)

        # Combine the labeled and unlabeled images along with the discriminative results.
        combined_batchX = np.concatenate((batch_image_s, batch_image_t))
        batch2Y = np.concatenate((batch_label, batch_label))
        combined_batchY = np.concatenate(
            (np.tile([0, 1], [batch_image_s.shape[0], 1]), np.tile([1, 0], [batch_image_t.shape[0], 1])))

        # return combined_batchX
        return combined_batchX,{'classifier_output': batch2Y, 'discriminator_output':combined_batchY}
        # return combined_batchX,batch2Y
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        batch_image, batch_label = self.get_batch_patches(index)
        if self.is_train:
            return batch_image, batch_label
        else:
            return batch_image
        # batch_image = self.get_batch_patches(index)
        # return batch_image


    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(self.patches_index.shape[0] / self.batch_size)

    def select_valid_sample(self):
        raw_patches_index = list(np.arange(0, self.num_patches))
        valid_patches_index = []
        for patch_idx in raw_patches_index:
            label = self._get_patch(self.labels_filenames, patch_idx)
            if not np.all(label == 0):
                valid_patches_index.append(patch_idx)
        self.patches_index = np.array(valid_patches_index)

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.patches_index)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.reset()

class SamplesDataLoaderNAS(SamplesDataLoader):
    def preprocess_input(self, img):
        img = img.astype('float32')
        img = preprocess_input(img)
        return img

class SamplesDataLoaderOtherBackbone(SamplesDataLoader):
    def preprocess_input(self, img):
        try:
            precess_fn = get_preprocessing(self.backbone)
        except:
            raise ValueError
        out = precess_fn(img)
        if np.max(out)>128.0:
            out = (out-128.0)/128.0
        return out

    def get_batch_patches(self, batch_index):
        batch_patch_idx = self.patches_index[batch_index *
                                             self.batch_size:(batch_index +
                                                              1) *
                                                             self.batch_size]

        batch_image = []
        batch_label = []
        for patch_idx in batch_patch_idx:
            image = self.get_patch(self.images_filenames, patch_idx)
            label = self.get_patch(self.labels_filenames, patch_idx)

            label[label > 0] = 1
            # 由于样本已经是被分割完毕小图，所以即使使用数据增强，也只能得到比原始图像更小范围内的数据，得不到更大范围的数据
            # if self.is_train:
            #     image, label = self.data_augmentation(image, label)
            label = self.preprocess_label(label)

            batch_image.append(image)
            batch_label.append(label)

        batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)

        batch_image = self.preprocess_input(batch_image)

        return batch_image, batch_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataloader = SamplesDataLoaderOtherBackbone(
        r'F:\Data\test\WnetTest\HR',
        r'F:\Data\test\WnetTest\GT',
        [256,256],
        [256,256],
        1,
        is_train=True,
        backbone='efficientnetb7',
        label_format='png'
    )
    # dataloader.on_epoch_end()
    itera = dataloader.__iter__()
    img,label = next(itera)

    fig,ax=plt.subplots(1,2)
    ax[0].imshow(img[0])
    ax[1].imshow(label[0].reshape((256,256,2))[:,:,0])
    plt.show()


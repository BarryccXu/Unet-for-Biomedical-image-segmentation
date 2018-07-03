# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 14:39:12 2017

@author: Chenchao Xu

Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html#motivation
"""
import numpy as np
import cv2
import os
from keras.utils import to_categorical
from utils import *

class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, dim_x=256, dim_y=256, dim_channel=1, dim_label=100, batch_size=10, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_channel = dim_channel
        self.dim_label = dim_label
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, list_IDs, image_dir=""):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp, image_dir)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_IDs_temp, image_dir=""):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y), dtype="float")
        y = np.empty((self.batch_size, self.dim_x, self.dim_y), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X_name = ID + "-img.npy"
            X_tmp_path = os.path.join(image_dir, X_name)
            img = np.load(X_tmp_path)
            img = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
            # Store class
            y_name = ID + "-seg.npy"
            y_tmp_path = os.path.join(image_dir, y_name)
            # for liver
            #seg = np.load(y_tmp_path)
            # for brain

            seg = np.load(y_tmp_path)
            seg = cv2.resize(seg, (self.dim_x, self.dim_y), interpolation=cv2.INTER_NEAREST)
            #y[i, ...] = label_map_dkt31_6(seg)
            y[i, ...] = label_map_dkt31_5(seg)
            X[i, ...] = norm_gray_matter(img, y[i,...])
            #y[i, ...] = label_map_3(seg, img)
            #X[i, ...] = norm_gray_white_matter(img, y[i, ...])

        return X[...,np.newaxis], to_categorical(y, num_classes=self.dim_label)




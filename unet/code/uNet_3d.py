# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from dataGeneratorKeras_3d import DataGenerator
import nibabel as nib
from utils import *
import cv2

'''
LABEL_NUM = 3
weights = [0.3, 0.6, 0.1] #3 labels
# DKT31 0: white matter, 1: background, 2~70: cortical labels
'''

LABEL_NUM = 14
#dkt31 =  [0.1603, 0.1787, 0.0384, 0.0446, 0.0215, 0.021, 0.0368, 0.034, 0.0712, 0.0639, 0.1479, 0.1814] # from 2d
#dkt31 = [0.1328, 0.1328, 0.0443, 0.0443, 0.021, 0.021, 0.0362, 0.0362, 0.0664, 0.0664, 0.1992, 0.1992] # from 2d modified
#weights = [0.0002, 0.0]
#dkt31 = [0.1604, 0.1779, 0.0384, 0.0447, 0.0216, 0.021, 0.0369, 0.034, 0.0709, 0.064, 0.1486, 0.181]
#weights = [0.0006, 0.0]
dkt31 = [0.1548, 0.1689, 0.044, 0.0457, 0.0224, 0.0222, 0.0349, 0.0347, 0.0779, 0.0724, 0.1476, 0.1738]
weights = [0.0007, 0.0]
weights.extend(dkt31)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #print intersection
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# define the weighted dice coef function
def dice_coef_weighted(y_true, y_pred, weights=weights):
    sum_loss = 0
    for i in range(0, len(weights)):
        sum_loss += weights[i] * dice_coef(y_true[..., i], y_pred[..., i])
    return sum_loss


def dice_coef_loss_weighted(y_true, y_pred, weights=weights):
    return 1.-dice_coef_weighted(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


class myUnet_3d(object):

    def __init__(self, model=None, save_name = '', img_x=64, img_y=64, img_z=64, batch_size=8, epoch=10, label_num=80):
        self.model = model
        self.img_x = img_x
        self.img_y = img_y
        self.img_z = img_z
        self.batch_size = batch_size
        self.epoch = epoch
        self.label_num = label_num
        self.save_name = save_name

    def generate_data_list_brain(self, prepared_data_name, label_type, num_patch = 64):
        train_list = []
        val_list = []
        for i in range(1, 20):
            tmp = get_list_patch_img_3d(prepared_data_name, label_type, i, num_patch=num_patch)
            if i <= 18:
                train_list.extend(tmp)
            elif i == 19:
                val_list.extend(tmp)
        return train_list, val_list

    def get_unet_3d(self):
        inputs = Input((self.img_x, self.img_y, self.img_z, 1))

        #    inp_norm = BatchNormalization()(inputs)

        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        '''
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        '''

        conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
        conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)

        '''
        up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        '''

        up7 = concatenate([UpSampling3D((2, 2, 2))(conv5), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)

        up8 = concatenate([UpSampling3D((2, 2, 2))(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)

        up9 = concatenate([UpSampling3D((2, 2, 2))(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)

        conv10 = Conv3D(self.label_num, (1, 1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss_weighted, metrics=[dice_coef_weighted])
        # default lr=1e-4
        self.model = model
        return model

    def train(self, prepared_data_name):
        print "generating data list"
        #train_list, val_list = self.generate_data_list_liver()
        train_list, val_list = self.generate_data_list_brain(prepared_data_name, self.save_name)
        print "Train data shape: ", len(train_list)
        print "Validation data shape: ", len(val_list)

        print"loading data done"
        model = self.get_unet_3d()
        print "got unet"
        params = {'dim_x': self.img_x,
                  'dim_y': self.img_y,
                  'dim_z': self.img_z,
                  'dim_label': self.label_num,
                  'batch_size': self.batch_size,
                  'shuffle': True}
        training_generator = DataGenerator(**params).generate(train_list)
        validation_generator = DataGenerator(**params).generate(val_list)
        model_save_name = 'unet_brain_3d_' + self.save_name + '.h5'
        model_checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=True)
        log_name = './logs/logs_unet_3d_' + self.save_name
        tensorboard = TensorBoard(log_dir=log_name)
        callbacks_list = [model_checkpoint, tensorboard]
        print 'Fitting model...'
        history_ft = model.fit_generator(
            generator=training_generator,
            steps_per_epoch=len(train_list) // self.batch_size,
            epochs=self.epoch,
            validation_data=validation_generator,
            validation_steps=len(val_list) // self.batch_size,
            class_weight='auto',
            callbacks=callbacks_list)

if __name__ == '__main__':
    config = {
        'save_name': 'stage_2',
        'img_x': 64,
        'img_y': 64,
        'img_z': 64,
        'label_num': LABEL_NUM,
        'epoch': 20,
        'batch_size': 16
    }
    myunet = myUnet_3d(**config)
    print '-'*60
    print 'Start training U-net...'
    myunet.train("prepared_3d")

    #prediction
    print '-'*60
    print 'Start doing prediction on test data...'

'''
for parameters reminder
dkt_6 label distribution of oasis-20-1
[0.8603757619857788,
 0.0034450888633728027,
 0.003105640411376953,
 0.014405429363250732,
 0.01237422227859497,
 0.025601208209991455,
 0.026294946670532227,
 0.014966726303100586,
 0.016230404376983643,
 0.007795572280883789,
 0.008634686470031738,
 0.0037174224853515625,
 0.0030528903007507324]
 
 [0.0006, 0.1604, 0.1779, 0.0384, 0.0447, 0.0216, 0.021, 0.0369, 0.034, 0.0709, 0.064, 0.1486, 0.181]

'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from dataGeneratorKeras import DataGenerator
'''
Ref: https://github.com/zhixuhao/unet/blob/master/unet.py
'''
smooth = 1.
# weights = [8.25089589e-06, 2.14682776e-02, 7.23189859e-03, 1.51260981e-02,
#    5.08206869e-02, 8.75058185e-03, 7.61365681e-03, 5.12948139e-03,
#    3.40352918e-02, 5.92399647e-03, 8.69393476e-03, 9.70766053e-03,
#    1.36333745e-02, 3.80576732e-03, 3.40635955e-02, 1.25783736e-02,
#    1.69017840e-02, 1.81727922e-02, 1.06269227e-02, 3.35887443e-02,
#    7.62286659e-03, 2.00840763e-02, 5.46262234e-03, 7.18559313e-03,
#    1.95287121e-02, 7.91296699e-03, 3.43449240e-03, 6.11593484e-03,
#    5.07608571e-03, 6.85777225e-03, 3.62972739e-02, 1.12903731e-02,
#    3.08793619e-02, 1.19351613e-02, 1.36379136e-02, 4.31627752e-02,
#    9.79236759e-03, 7.81334738e-03, 5.94765117e-03, 4.30267580e-02,
#    5.05728423e-03, 8.44392365e-03, 9.66414383e-03, 2.10978489e-02,
#    5.07514231e-03, 4.72723296e-02, 1.64009904e-02, 1.10363663e-02,
#    4.77406452e-02, 1.50676747e-02, 2.86143721e-02, 7.35394500e-03,
#    2.14065710e-02, 4.80853127e-03, 7.69012928e-03, 2.38425341e-02,
#    6.13471224e-03, 2.57943789e-03, 6.09137834e-03, 4.86854147e-03,
#    4.74008837e-03, 6.27762048e-02, 1.12919293e-02]
#label_num = len(weights)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # define the weighted dice coef function


def dice_coef_weight(y_true, y_pred):
    sum_loss = 0
    for i in range(0, label_num):
        sum_loss += weights[i] * dice_coef(y_true[..., i], y_pred[..., i])
    return sum_loss


def dice_coef_loss_weight(y_true, y_pred):
    return -dice_coef_weight(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class myUnet(object):

    def __init__(self, img_rows=256, img_cols=256, batch_size = 10, epoch = 2, label_num = 80):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.epoch = epoch
        self.label_num = label_num

    def generate_data_list(self):
        data_set = "OASIS-TRT-20"
        data_path = "../image/prepared/train/"
        num_img = 20
        train_list = []
        val_list = []
        for i in range(1, 20):
            for j in range(0, 160):
                file_name = data_path + data_set + "-" + str(i) + "-" + str(j).zfill(3)
                if i <= 18:
                    train_list.append(file_name)
                elif i == 19:
                    val_list.append(file_name)
        return train_list, val_list

    def generate_data_list_liver(self):
        data_set = "liver"
        data_path = "../image/prepared_liver/train/"
        num_img = 20
        train_list = []
        val_list = []
        for i in range(1, 20):
            for j in range(0, 128):
                file_name = data_path + data_set + "-" + str(i).zfill(3) + "-" + str(j).zfill(3)
                if i <= 18:
                    train_list.append(file_name)
                elif i == 19:
                    val_list.append(file_name)
        return train_list, val_list

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        #    inp_norm = BatchNormalization()(inputs)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(self.label_num, (1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss_weight, metrics=[dice_coef])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        print "generating data list"
        train_list, val_list = self.generate_data_list_liver()
        print "Train data shape: ", len(train_list)
        print "Validation data shape: ", len(val_list)

        print"loading data done"
        model = self.get_unet()
        print "got unet"
        params = {'dim_x': self.img_rows,
                  'dim_y': self.img_cols,
                  'dim_channel': 1,
                  'dim_label': self.label_num,
                  'batch_size': self.batch_size,
                  'shuffle': True}
        training_generator = DataGenerator(**params).generate(train_list)
        validation_generator = DataGenerator(**params).generate(val_list)
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print 'Fitting model...'
        history_ft = model.fit_generator(
            generator=training_generator,
            steps_per_epoch=len(train_list) // self.batch_size,
            epochs=self.epoch,
            validation_data=validation_generator,
            validation_steps=len(val_list) // self.batch_size,
            class_weight='auto')
        model.save("model_liver-epoch_2.kerasmodel")
        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    # def save_img(self):
    #     print("array to image")
    #     imgs = np.load('imgs_mask_test.npy')
    #     for i in range(imgs.shape[0]):
    #         img = imgs[i]
    #         img = array_to_img(img)
    #         img.save("../results/%d.jpg" % (i))

if __name__ == '__main__':
    myunet = myUnet(img_cols=128, img_rows=128, label_num=2)
    myunet.train()
    #myunet.save_img()
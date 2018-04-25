import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from dataGeneratorKeras import DataGenerator
import nibabel as nib
'''
Ref: https://github.com/zhixuhao/unet/blob/master/unet.py
'''
weights = [0.1, 0.9]

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # define the weighted dice coef function


def dice_coef_weight(y_true, y_pred, weights=weights):
    sum_loss = 0
    for i in range(0, len(weights)):
        sum_loss += weights[i] * dice_coef(y_true[..., i], y_pred[..., i])
    return sum_loss


def dice_coef_loss_weight(y_true, y_pred, weights=weights):
    return -dice_coef_weight(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class myUnet(object):

    def __init__(self, model=None, img_rows=256, img_cols=256, batch_size = 96, epoch = 2, label_num = 80):
        self.model = model
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.epoch = epoch
        self.label_num = label_num

    def generate_data_list_brain(self):
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

        model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss_weight, metrics=[dice_coef])
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self):
        print "generating data list"
        #train_list, val_list = self.generate_data_list_liver()
        train_list, val_list = self.generate_data_list_brain()
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
        #model_checkpoint = ModelCheckpoint('unet.h5', monitor='loss', verbose=1, save_best_only=True)
        model_checkpoint = ModelCheckpoint('unet_brain.h5', monitor='loss', verbose=1, save_best_only=True)
        print 'Fitting model...'
        history_ft = model.fit_generator(
            generator=training_generator,
            steps_per_epoch=len(train_list) // self.batch_size,
            epochs=self.epoch,
            validation_data=validation_generator,
            validation_steps=len(val_list) // self.batch_size,
            class_weight='auto',
            callbacks=[model_checkpoint])
        #model.save("model_liver-epoch_2.kerasmodel")
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
def predict_brain(model, weights):
    model.load_weights(weights)
    imgs = np.ndarray((160,256,256), dtype="float")
    segs = np.ndarray((160,256,256), dtype = int)
    for i in range(0, 160):
        img_name = "OASIS-TRT-20-20-" + str(i).zfill(3) + "-img.npy"
        seg_name = "OASIS-TRT-20-20-" + str(i).zfill(3) + "-seg.npy"
        img_path = os.path.join("../image/prepared/test", img_name)
        seg_path = os.path.join("../image/prepared/test", seg_name)
        imgs[i,:,:] = np.load(img_path)
        segs[i,:,:] = np.load(seg_path)
    preds = model.predict(imgs[..., np.newaxis])
    img_pred = np.argmax(preds, axis=3)
    # for 0-1 segmentation
    np.save("pred_brain-20.npy", preds)
    segs[segs != 0] = 1
    dic = dice_coef_2class(img_preds, segs)
    # seg_shape = img_pred.shape
    # seg_1d = img_pred.flatten()
    # for i, val in enumerate(seg_1d):
    #     if 0 < val <= 35:
    #         seg_1d[i] = val + 1000
    #     elif val > 35:
    #         seg_1d[i] = val + 2000 - 35
    # seg = seg_1d.reshape(seg_shape)
    # np.save("results", seg)
    return dic

def predict_liver(model, weights):
    #model = load_model(model, custom_objects={'loss': dice_coef_loss_weight, 'metrics': dice_coef})
    model.load_weights(weights)
    imgs = np.ndarray((128, 128, 128), dtype="float")
    segs = np.ndarray((128, 128, 128), dtype=int)
    for i in range(0, 128):
        img_name = "liver-020-" + str(i).zfill(3) + "-img.npy"
        seg_name = "liver-020-" + str(i).zfill(3) + "-seg.npy"
        img_path = os.path.join("../image/prepared_liver/test", img_name)
        seg_path = os.path.join("../image/prepared_liver/test", seg_name)
        imgs[i,:,:] = np.load(img_path)
        segs[i,:,:] = np.load(seg_path)
    preds = model.predict(imgs[..., np.newaxis])
    img_preds = np.argmax(preds, axis=3)
    #save 3d image
    #pair_img = nib.Nifti1Pair(img_preds, np.eye(4))
    #nib.save(pair_img, "pred_liver-20.img")
    np.save('pred_liver-20.npy', img_preds)
    #compute dice coefficient
    dic = dice_coef_2class(img_preds, segs)
    return dic

def dice_coef_2class(pred, ground):
    n_p = 0
    n_g = 0
    n_pg = 0
    #print pred[64,]
    #print ground[64,]
    pred = pred.flatten()
    ground = ground.flatten()
    for i in range(0, len(pred)):
        if ground[i] == 1 and pred[i] == 1:
            n_pg += 1
        n_p += pred[i]
        n_g += ground[i]
    return 2*float(n_pg) / float(n_p + n_g)

if __name__ == '__main__':
    #myunet = myUnet(img_cols=128, img_rows=128, label_num=2, epoch=10) #for liver
    myunet = myUnet(img_cols=256, img_rows=256, label_num=2, epoch=10, batch_size=48)
    print '-'*30
    print 'Start training U-net...'
    myunet.train()
    #prediction
    print '-'*30
    print 'Start doing prediction on test data...'
    dic = predict_brain(myunet.model, 'unet_brain.h5')
    print 'Dice of Test data:'
    print dic
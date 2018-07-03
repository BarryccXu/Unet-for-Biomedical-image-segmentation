import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from dataGeneratorKeras import DataGenerator
import nibabel as nib
from utils import *
import cv2
'''
Ref: https://github.com/zhixuhao/unet/blob/master/unet.py
'''
# 13 labels
# label_stat = [0.95, 0.0015, 0.0015, 0.0045, 0.0045, 0.0095, 0.0095, 0.0055, 0.0055, 0.003, 0.003, 0.001, 0.001]
'''
LABEL_NUM = 3
weights = [0.3, 0.6, 0.1] #3 labels
# DKT31 0: white matter, 1: background, 2~70: cortical labels,
'''
'''
LABEL_NUM = 14
#dkt31 =  [0.1603, 0.1787, 0.0384, 0.0446, 0.0215, 0.021, 0.0368, 0.034, 0.0712, 0.0639, 0.1479, 0.1814]
dkt31 = [0.1328, 0.1328, 0.0443, 0.0443, 0.021, 0.021, 0.0362, 0.0362, 0.0664, 0.0664, 0.1992, 0.1992]
weights = [0.0002, 0.0]
weights.extend(dkt31)
'''
LABEL_NUM = 12
dkt31 = [0.1] * 10
weights = [0.0, 0.0]
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

class myUnet(object):

    def __init__(self, model=None, save_name = '', img_rows=256, img_cols=256, batch_size = 96, epoch = 2, label_num = 80):
        self.model = model
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.epoch = epoch
        self.label_num = label_num
        self.save_name = save_name

    def generate_data_list_brain(self, prepared_data):
        data_set = "OASIS-TRT-20"
        data_path = os.path.join("../image/", prepared_data, "train/")
        num_img = 20
        train_list = []
        val_list = []
        for i in range(1, 20):
            for j in range(0, 160):
                file_name = data_path + data_set + "-" + str(i) + "-" + str(j).zfill(3)
                file_name_check = file_name + "-img.npy"
                if os.path.exists(file_name_check):
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

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)

        up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)

        up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)

        up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)

        conv10 = Conv2D(self.label_num, (1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        #model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001), loss=dice_coef_loss_weight, metrics=[dice_coef_weight])
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss_weighted, metrics=[dice_coef_weighted])
        self.model = model
        return model

    def train(self, prepared_data_name):
        print "generating data list"
        #train_list, val_list = self.generate_data_list_liver()
        train_list, val_list = self.generate_data_list_brain(prepared_data_name)
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
        model_save_name = 'unet_brain_' + self.save_name + '.h5'
        model_checkpoint = ModelCheckpoint(model_save_name, monitor='loss', verbose=1, save_best_only=True)
        log_name = './logs/logs_unet_' + self.save_name
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

def predict_brain(model, weights, dataset_type, slide_type):
    model.load_weights(weights)
    pre = ''
    if dataset_type == "val":
        dataset_type = "train"
        pre = pre.join("OASIS-TRT-20-19-")
    elif dataset_type == "test":
        pre = pre.join("OASIS-TRT-20-20-")
    else:
        raise ValueError("The data_type is wrong: {}".format(dataset_type))
    result_name = pre + "pred.npy"
    if slide_type == "sagittal":
        imgs = np.ndarray((160, 256, 256), dtype="float")
        segs = np.ndarray((160, 256, 256), dtype=int)
        save_path = os.path.join("../image/results_sagittal/", result_name)
        for i in range(0, 160):
            img_name = pre + str(i).zfill(3) + "-img.npy"
            seg_name = pre + str(i).zfill(3) + "-seg.npy"
            img_path = os.path.join("../image/prepared/", dataset_type, img_name)
            seg_path = os.path.join("../image/prepared/", dataset_type, seg_name)
            imgs[i,...] = np.load(img_path)
            segs[i,...] = np.load(seg_path)
            segs[i,...] = label_map_dkt31(segs[i, ...])
            imgs[i,...] = norm_gray_matter(imgs[i,...], segs[i,...])
    elif slide_type == 'horizontal':
        imgs = np.ndarray((256, 256, 256), dtype="float")
        segs = np.ndarray((256, 256, 256), dtype=int)
        save_path = os.path.join("../image/results_horizontal/", result_name)
        for i in range(0, 256):
            img_name = pre + str(i).zfill(3) + "-img.npy"
            seg_name = pre + str(i).zfill(3) + "-seg.npy"
            img_path = os.path.join("../image/prepared_2d_horizontal/", dataset_type, img_name)
            seg_path = os.path.join("../image/prepared_2d_horizontal/", dataset_type, seg_name)
            if os.path.exists(img_path):
                img = np.load(img_path)
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
                seg = np.load(seg_path)
                seg = cv2.resize(seg, (256, 256), interpolation=cv2.INTER_NEAREST)
                #segs[:,i,:] = label_map_dkt31_6(seg)
                segs[:,i,:] = label_map_dkt31_5(seg)
                imgs[:,i,:] = norm_gray_matter(img, segs[:,i,:])
                #segs[:, i, :] = label_map_3(seg, img)
                #imgs[:, i, :] = norm_gray_white_matter(img, segs[:, i, :])
    preds = model.predict(imgs[..., np.newaxis])
    img_preds = np.argmax(preds, axis=3)
    dice = dice_dkt_6(img_preds, segs, LABEL_NUM)
    np.save(save_path, img_preds)
    return dice

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
        imgs[i,...] = np.load(img_path)
        segs[i,...] = np.load(seg_path)
    preds = model.predict(imgs[..., np.newaxis])
    img_preds = np.argmax(preds, axis=3)
    dic = 0.
    #save 3d image
    #pair_img = nib.Nifti1Pair(img_preds, np.eye(4))
    #nib.save(pair_img, "pred_liver-20.img")
    #compute dice coefficient
    #dic = dice_coef_2class(img_preds, segs)

    np.save('pred_liver-20.npy', img_preds)
    #compute accuracy

    return dic


if __name__ == '__main__':
    #myunet = myUnet(img_cols=128, img_rows=128, label_num=2, epoch=10) #for liver
    #myunet = myUnet(img_cols=256, img_rows=256, label_num=3, epoch=30, batch_size=32)
    #myunet = myUnet(save_name = 'stage_1', img_cols=256, img_rows=256, label_num=LABEL_NUM, epoch=30, batch_size=32)
    myunet = myUnet(save_name = 'stage_2', img_cols=256, img_rows=256, label_num=LABEL_NUM, epoch=10, batch_size=8)
    #myunet = myUnet(img_cols=256, img_rows=256, label_num=36, epoch=5, batch_size=16)

    print '-'*60
    print 'Start training U-net...'
    myunet.train("prepared_2d_horizontal")

    #prediction
    myunet.get_unet()
    print '-'*60
    print 'Start doing prediction on test data...'
    dice = predict_brain(myunet.model, "unet_brain_stage_2.h5", "test", 'horizontal')
    print "Dice: ", dice

    '''
    results
    Dice:  [0.9888322147137434, 0.0, 0.45207414942918506, 0.43235545428652805, 0.5822206354873259, 0.0, 0.0, 0.5154851783440633, 0.05502204669029134, 0.1904094541873414, 0.0, 0.1507121024907329, 0.4460803306212063, 0.5218681879319141]
    '''

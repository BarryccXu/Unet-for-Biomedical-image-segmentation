import os
import numpy as np
import nibabel as nib
import h5py
from utils import *

def prepare_brain_2d_sagittal(file_name, dataset_type):
    data_dir = "../image/OASIS-TRT-20_volumes/"
    save_dir = "../image/prepared/"
    img, seg = load_brain_3d(file_name, data_dir)
    # preprocess data
    #img = (img - IMG_MEAN) / IMG_STD
    # save data
    save_path = os.path.join(save_dir, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(0, img.shape[2]):
        img_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-img" + ".npy"
        seg_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-seg" + ".npy"
        np.save(img_save_path, img[:,:,i])
        np.save(seg_save_path, seg[:,:,i])

def prepare_brain_2d_horizontal(file_name, dataset_type):
    data_dir = "../image/OASIS-TRT-20_volumes/"
    save_dir = "../image/prepared_2d_horizontal/"
    print file_name
    img, seg = load_brain_3d(file_name, data_dir)
    save_path = os.path.join(save_dir, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(0, img.shape[1]):
        if np.sum(img[:,i,:]) == 0.0:
            continue
        else:
            img_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-img" + ".npy"
            seg_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-seg" + ".npy"
            np.save(img_save_path, img[:, i, :])
            np.save(seg_save_path, seg[:, i, :])

def prepare_data_liver(data_dir):
    save_dir_train = "../image/prepared_liver/train/"
    save_dir_test = "../image/prepared_liver/test/"
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    import skimage.io
    num_img = 20
    for i in range(1, num_img + 1):
        if i <= 19:
            save_dir = save_dir_train
        else:
            save_dir = save_dir_test
        img_file = "small-orig" + str(i).zfill(3) + ".mhd"
        seg_file = "small-seg" + str(i).zfill(3) + ".mhd"
        img = skimage.io.imread(os.path.join(data_dir, img_file), plugin='simpleitk')
        seg = skimage.io.imread(os.path.join(data_dir, seg_file), plugin='simpleitk')
        img_shape = img.shape
        for j in range(0, img_shape[2]):
            img_save_path = save_dir + "/" + "liver-" + str(i).zfill(3) + \
                            "-" + str(j).zfill(3) + "-img" + ".npy"
            seg_save_path = save_dir + "/" + "liver-" + str(i).zfill(3) + \
                            "-" + str(j).zfill(3) + "-seg" + ".npy"
            np.save(img_save_path, img[:, :, j])
            np.save(seg_save_path, seg[:, :, j])

def load_brain_3d(file_name, data_dir):
    # load data
    # file_name = img_set + '-' + img_idx
    img_dir = os.path.join(data_dir, file_name, "t1weighted_brain.nii.gz")
    img = nib.load(img_dir)
    seg_dir = os.path.join(data_dir, file_name, "labels.DKT31.manual.nii.gz")
    seg = nib.load(seg_dir)

    # convert to numpy ndarray
    img = img.get_data()  # dtype = '<i2'
    img = img.astype('float')
    seg = seg.get_data()  # dtype = '<f4'
    return img, seg

if __name__ == '__main__':
    # brain
    data_set = "OASIS-TRT-20"
    num_img = 20
    for i in range(1, num_img):
        print i
        file_train = data_set + "-" + str(i)
        prepare_brain_2d_horizontal(file_train, "train")
    file_test = data_set + "-" + str(num_img)
    prepare_brain_2d_horizontal(file_test, "test")

    #prepare_data_liver("../image/liver_2007/")

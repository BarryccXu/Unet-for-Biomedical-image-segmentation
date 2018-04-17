import os
import numpy as np
import nibabel as nib
import h5py
from utils import *

IMG_MEAN = 160.
IMG_STD = 429.

def prepare_data(file_name, dataset_type, label_map = dict()):
    # load data
    #file_name = img_set + '-' + img_idx
    data_dir = "../image/OASIS-TRT-20_volumes/"
    save_dir = "../image/prepared/"
    img_dir = os.path.join(data_dir, file_name, "t1weighted_brain.nii.gz")
    img = nib.load(img_dir)
    seg_dir = os.path.join(data_dir, file_name, "labels.DKT31.manual.nii.gz")
    seg = nib.load(seg_dir)

    # convert to numpy ndarray
    img = img.get_data() # dtype = '<i2'
    img = img.astype('float')
    seg = seg.get_data() # dtype = '<f4'

    # preprocess data
    img = (img - IMG_MEAN) / IMG_STD
    u, c = np.unique(seg, return_counts=True)
    for i in range(len(u)):
        if u[i] in label_map:
            label_map[u[i]] += c[i]
        else:
            label_map[u[i]] = c[i]
    # save data
    save_path = os.path.join(save_dir, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(0, img.shape[2]):
        img_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-img" + ".npy"
        seg_save_path = save_path + "/" + file_name + "-" + str(i).zfill(3) + "-seg" + ".npy"
        np.save(img_save_path, img[:,:,i])
        np.save(seg_save_path, seg[:,:,i])
    return label_map

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




if __name__ == "__main__":
    # brain
    # data_set = "OASIS-TRT-20"
    # num_img = 20
    # label_count = dict()
    # for i in range(1, num_img):
    #     print i
    #     file_train = data_set + "-" + str(i)
    #     label_count = prepare_data(file_train, "train", label_count)
    # np.save('label_count_1-19',label_count)
    #
    # file_test = data_set + "-" + str(num_img)
    # prepare_data(file_test, "test")
    prepare_data_liver("../image/liver_2007/")

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import cv2
import os
from utils import *


DIM_X = 256
DIM_Y = 256
DIM_Z = 160
PATCH_SIZE = 64

def remove_border(img_3d, seg_3d):
    x1=DIM_X; x2=0; y1=DIM_Y; y2=0; z1=DIM_Z; z2=0
    for i in range(0, DIM_X):
        for j in range(0, DIM_Y):
            for k in range(0, DIM_Z):
                if img_3d[i, j, k] != 0:
                    x1 = min(x1, i)
                    x2 = max(x2, i)
                    y1 = min(y1, j)
                    y2 = max(y2, j)
                    z1 = min(z1, k)
                    z2 = max(z2, k)
    return img_3d[x1-2:x2+3, y1-2:y2+3, z1-2:z2+3], seg_3d[x1-2:x2+3, y1-2:y2+3, z1-2:z2+3]

def resize_to_256(img_3d):
    dim = img_3d.shape
    return zoom(img_3d, (256/float(dim[0]), 256/float(dim[1]), 256/float(dim[2])), order=0) #order 0 means nearest interpolation

def img_to_patch(img_3d, dim):
    imgs_patch = []
    orig_dim = img_3d.shape[0]
    axis = range(0, orig_dim + 1, dim)
    for x in range(len(axis)-1):
        for y in range(len(axis)-1):
            for z in range(len(axis)-1):
                img_patch = img_3d[axis[x]:axis[x+1], axis[y]:axis[y+1], axis[z]:axis[z+1]]
                imgs_patch.append(img_patch)
    return imgs_patch

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

def get_label_ditrib_3d(file_name, label_type):
    data_dir = "../image/OASIS-TRT-20_volumes/"
    img, seg = load_brain_3d(file_name, data_dir)
    # get ROI by removing border
    img, seg = remove_border(img, seg)
    # rescale the image to 256x256x256 with nearest interpolation
    img = resize_to_256(img)
    seg = resize_to_256(seg)
    # relabel
    if label_type == "stage_1":
        seg = label_map_3(seg, img)
    elif label_type == "stage_2":
        seg = label_map_dkt31_6(seg)
    else:
        raise ValueError("Wrong label_type value!")
    u, c = np.unique(seg, return_counts=True)
    return u, c

def get_label_ditrib_3d_all():
    '''
    :param label_type: "stage_1" or "stage_2"
    :return:
    '''
    '''
    results on 1-19
    d1 = [0.21246825080168874, 0.13757515581030594, 0.6499565933880053]
    d2 = [0.8624252394626015, 0.0036507437103673033, 0.003346142015959087, 0.01284337043762207, 0.012378520087191933, 0.025204373033423173, 0.025502148427461322, 0.016202484306536223, 0.016304282765639454, 0.00725322961807251, 0.007807687709206029, 0.003828898856514379, 0.003252879569405004]
    '''
    # brain
    data_set = "OASIS-TRT-20"
    num_img = 20
    s1 = [0.0]*3
    s2 = [0.0]*13
    for i in range(1, num_img):
        print "Processing image No.", str(i), "of ", data_set, "........."
        file_train = data_set + "-" + str(i)
        _, c1 = get_label_ditrib_3d(file_train, "stage_1")
        s1 = [(x + y) for x, y in zip(s1, c1)]
        _, c2 = get_label_ditrib_3d(file_train, "stage_2")
        s2 = [(x + y) for x, y in zip(s2, c2)]
    sum1 = float(sum(s1))
    distrib1 = map(lambda x : x/ sum1, s1)
    sum2 = float(sum(s2))
    distrib2 = map(lambda x : x / sum2, s2)
    return distrib1, distrib2

def distrib_to_weights(d):
    '''
    results on 1-19
    w1 = [0.3483, 0.5379, 0.1139]
    w2 = [0.0007, 0.1548, 0.1689, 0.044, 0.0457, 0.0224, 0.0222, 0.0349, 0.0347, 0.0779, 0.0724, 0.1476, 0.1738]

    '''
    div = map(lambda x : 1/x, d)
    s = sum(div)
    w = map(lambda x : round(x/s, 4), div)
    return w



def prepare_brain_3d(file_name, dataset_type, label_type):
    data_dir = "../image/OASIS-TRT-20_volumes/"
    save_dir = "../image/prepared_3d/" + label_type
    #print file_name
    img, seg = load_brain_3d(file_name, data_dir)
    save_path = os.path.join(save_dir, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get ROI by removing border
    img, seg = remove_border(img, seg)
    # rescale the image to 256x256x256 with nearest interpolation
    img = resize_to_256(img)
    seg = resize_to_256(seg)
    # relabel & do normalization
    if label_type == "stage_1":
        seg = label_map_3(seg, img)
        img = norm_gray_white_matter(img, seg)
    elif label_type == "stage_2":
        seg = label_map_dkt31_6(seg)
        img = norm_gray_matter(img, seg)
    else:
        raise ValueError("Wrong label_type value!")
    imgs_patch = img_to_patch(img, PATCH_SIZE)
    segs_patch = img_to_patch(seg, PATCH_SIZE)
    for i in range(len(imgs_patch)):
        img_save_path = save_path + "/" + file_name + "-" + str(i).zfill(2) + "-img" + ".npy"
        seg_save_path = save_path + "/" + file_name + "-" + str(i).zfill(2) + "-seg" + ".npy"
        np.save(img_save_path, imgs_patch[i])
        np.save(seg_save_path, segs_patch[i])


if __name__ == '__main__':
    # brain
    data_set = "OASIS-TRT-20"
    num_img = 20
    for i in range(1, num_img):
        print "Processing image No.", str(i), "of ", data_set, "to", "train & validation set"
        file_train = data_set + "-" + str(i)
        prepare_brain_3d(file_train, "train", "stage_1")
        prepare_brain_3d(file_train, "train", "stage_2")
    file_test = data_set + "-" + str(num_img)
    print "Processing image No.", str(num_img), "of ", data_set, "to", "test set"
    prepare_brain_3d(file_test, "test", "stage_1")
    prepare_brain_3d(file_test, "test", "stage_2")

    '''
    #for debug
    img = nib.load("/home/caffe/keras_tensorflow/image/Unet-for-Biomedical-image-segmentation/unet/image/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
    seg = nib.load("/home/caffe/keras_tensorflow/image/Unet-for-Biomedical-image-segmentation/unet/image/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
    img = img.get_data()
    seg = seg.get_data()
    img_n, seg_n = remove_border(img, seg)
    cv2.imwrite("seg_3d_x100.jpg", seg_n[100, ...])
    seg_n = resize_to_256(seg_n)
    cv2.imwrite("seg_3d_x100_zoom.jpg", seg_n[100,...])
    '''



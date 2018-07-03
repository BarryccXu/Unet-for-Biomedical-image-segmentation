import numpy as np
import nibabel as nib
import os

IMG_MEAN = 160.
IMG_STD = 429.
G_MEAN = 1397.
G_STD = 296.
GW_MEAN = 1222.
GW_STD = 323.
DATA_SET = "OASIS-TRT-20"

def label_map_dkt31(seg_2d):
    # works for 3d input image
    # 0: white matter & background, 2~69: cortical labels,
    seg_shape = seg_2d.shape
    seg_1d = seg_2d.flatten()
    for idx, val in enumerate(seg_1d):
        val = int(val)
        if val == 0:
            continue
        elif 1002 <= val <= 1035:
            seg_1d[idx] = val - 1000
            #seg_1d[idx] = 2
        elif 2002 <= val <= 2035:
            seg_1d[idx] = val - 2000 + 34
            #seg_1d[idx] = 3
        else:
            raise ValueError("Could not convert label number {} for brain segmentation".format(val))
    seg_2d = seg_1d.reshape(seg_shape)
    #seg_2d[img_2d == (0. - IMG_MEAN) / IMG_STD] = 1  # background
    #img_2d[seg_2d == 0] = (0. - IMG_MEAN) / IMG_STD
    return seg_2d

def label_map_dkt31_6(seg_2d):
    # works for 3d input image
    # 0: white matter & background
    # 2,3: TEMPORAL_LOBE_MEDIAL
    TEMPORAL_LOBE_MEDIAL = {6, 16, 7}
    # 4,5: TEMPORAL_LOBE_LATERAL
    TEMPORAL_LOBE_LATERAL = {30, 15, 9, 34}
    # 6,7: FRONTAL_LOBE
    FRONTAL_LOBE = {28, 12, 14, 24, 17, 3, 18, 19, 20, 27}
    # 8,9: PARIETAL_LOBE
    PARIETAL_LOBE = {22, 31, 29, 8, 25}
    # 10, 11: OCCIPITAL_LOBE
    OCCIPITAL_LOBE = {13, 21, 5, 11}
    # 12, 13: CINGULATE_CORTEX
    CINGULATE_CORTEX = {10, 23, 26, 2}
    # removed #35
    label_cluster = [TEMPORAL_LOBE_MEDIAL, TEMPORAL_LOBE_LATERAL, FRONTAL_LOBE, PARIETAL_LOBE, OCCIPITAL_LOBE, CINGULATE_CORTEX]
    def get_new_label(old_label, left_right):
        for idx, lobe in enumerate(label_cluster):
            if old_label in lobe:
                if left_right == "left":
                    return 2+2*idx
                elif left_right == "right":
                    return 3+2*idx
                else:
                    raise ValueError("Wrong left_right value!")
        return 0

    seg_shape = seg_2d.shape
    seg_1d = seg_2d.flatten()
    for idx, val in enumerate(seg_1d):
        val = int(val)
        if val == 0:
            continue
        elif 1002 <= val <= 1035:
            seg_1d[idx] = get_new_label(val-1000, "left")
        elif 2002 <= val <= 2035:
            seg_1d[idx] = get_new_label(val-2000, "right")
        else:
            raise ValueError("Could not convert label number {} for brain segmentation".format(val))
    seg_2d = seg_1d.reshape(seg_shape)
    return seg_2d

def label_map_dkt31_5(seg_2d):
    # works for 3d input image
    # 0: white matter & background
    # 2,3: TEMPORAL_LOBE
    TEMPORAL_LOBE = {6, 16, 7, 30, 15, 9, 34}
    # 4,5: FRONTAL_LOBE
    FRONTAL_LOBE = {28, 12, 14, 24, 17, 3, 18, 19, 20, 27}
    # 6,7: PARIETAL_LOBE
    PARIETAL_LOBE = {22, 31, 29, 8, 25}
    # 8,9: OCCIPITAL_LOBE
    OCCIPITAL_LOBE = {13, 21, 5, 11}
    # 10,11: CINGULATE_CORTEX
    CINGULATE_CORTEX = {10, 23, 26, 2}
    # removed #35
    label_cluster = [TEMPORAL_LOBE, FRONTAL_LOBE, PARIETAL_LOBE, OCCIPITAL_LOBE, CINGULATE_CORTEX]
    def get_new_label(old_label, left_right):
        for idx, lobe in enumerate(label_cluster):
            if old_label in lobe:
                if left_right == "left":
                    return 2+2*idx
                elif left_right == "right":
                    return 3+2*idx
                else:
                    raise ValueError("Wrong left_right value!")
        return 0

    seg_shape = seg_2d.shape
    seg_1d = seg_2d.flatten()
    for idx, val in enumerate(seg_1d):
        val = int(val)
        if val == 0:
            continue
        elif 1002 <= val <= 1035:
            seg_1d[idx] = get_new_label(val-1000, "left")
        elif 2002 <= val <= 2035:
            seg_1d[idx] = get_new_label(val-2000, "right")
        else:
            raise ValueError("Could not convert label number {} for brain segmentation".format(val))
    seg_2d = seg_1d.reshape(seg_shape)
    return seg_2d

def label_map_3(seg_2d, img_2d):
    # works for 3d input image
    # 0 is white matter
    seg_2d[seg_2d == 1035] = 0 # set the insula as background
    seg_2d[seg_2d == 2035] = 0
    seg_2d[seg_2d != 0] = 1  # gray matter
    seg_2d[img_2d == 0] = 2  # background
    return seg_2d

def label_map(seg_dir):
    seg = nib.load(seg_dir)
    seg = seg.get_data()
    unique_num = np.unique(seg)
    map = dict()
    for idx, val in enumerate(unique_num):
        map[val] = idx
    return map

def get_mean_value_of_graywhite_matter(img_3d):
    vals = []
    for itr in np.nditer(img_3d):
        val = itr.item()
        if val != 0:
            vals.append(val)
    return np.mean(vals), np.std(vals)

def get_mean_value_of_gray_matter(img_3d, seg_3d):
    vals = []
    for itr_1, itr_2 in np.nditer([img_3d, seg_3d]):
        img = itr_1.item()
        seg = itr_2.item()
        if img != 0 and seg == 0:
            vals.append(img)
    return np.mean(vals), np.std(vals)

def norm_gray_matter(img_2d, seg_2d):
    # works for 3d input image
    img_2d[seg_2d == 0] = G_MEAN
    img_2d -= G_MEAN
    img_2d /= G_STD
    return img_2d

def norm_gray_white_matter(img_2d, seg_2d):
    # works for 3d input image
    img_2d[seg_2d == 0] = GW_MEAN
    img_2d -= GW_MEAN
    img_2d /= GW_STD
    return img_2d

def get_list_patch_img_3d(prepared_data_name, label_type, img_num, num_patch = 64):
    data_path = os.path.join("../image/", prepared_data_name, label_type, "train/")
    patches = []
    for i in range(0, num_patch):
        file_name = data_path + DATA_SET + "-" + str(img_num) + "-" + str(i).zfill(2)
        file_name_check = file_name + "-img.npy"
        if os.path.exists(file_name_check):
            patches.append(file_name)
    return patches

def dice_dkt_6(pre, ground, num_label):
    #14 labels
    smooth = 1.
    pre_f = pre.flatten()
    ground_f = ground.flatten()
    n_p = [0.] * num_label
    n_g = [0.] * num_label
    n_pg = [0.] * num_label
    for i in range(0, len(pre_f)):
        n_p[int(pre_f[i])] += 1
        n_g[int(ground_f[i])] += 1
        if pre_f[i] == ground_f[i]:
            n_pg[int(pre_f[i])] += 1
    return [2*pg / (p+g+smooth) for pg, p, g in zip(n_pg, n_p, n_g)]

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

if __name__ == "__main__":
    map = label_map("/home/caffe/keras_tensorflow/image/"
                    "Unet-for-Biomedical-image-segmentation/unet/image/OASIS-TRT-20_volumes"
                    "/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")

import numpy as np
import nibabel as nib

def label_map(seg_dir):
    seg = nib.load(seg_dir)
    seg = seg.get_data()
    unique_num = np.unique(seg)
    map = dict()
    for idx, val in enumerate(unique_num):
        map[val] = idx
    return map

def get_unique_values():



if __name__ == "__main__":
    map = label_map("/home/caffe/keras_tensorflow/image/"
                    "Unet-for-Biomedical-image-segmentation/unet/image/OASIS-TRT-20_volumes"
                    "/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")

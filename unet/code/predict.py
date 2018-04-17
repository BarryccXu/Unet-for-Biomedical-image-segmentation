from keras.models import load_model
import numpy as np
import os
import nibabel as nib
import skimage.io
def predict(model):
    model = load_model(model)
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

    seg_shape = img_pred.shape
    seg_1d = img_pred.flatten()
    for i, val in enumerate(seg_1d):
        if 0 < val <= 35:
            seg_1d[i] = val + 1000
        elif val > 35:
            seg_1d[i] = val + 2000 - 35
    seg = seg_1d.reshape(seg_shape)
    np.save("results", seg)
    #img_save = nib.Nifti1Image(img_pre)
    #nib.save(img_save, "prediction.nii.gz")

def predict_liver(model):
    model = load_model(model)
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
    img_pred = np.argmax(preds, axis=3)
    #np.save("results_liver", img_pred)
    #img_save = nib.Nifti1Image(img_pred, np.eye(4))
    #nib.save(img_save, "pred_liver-20.nii.gz")
    skimage.io.imsave("Pred_liver-20", img_pred, plugin='simpleitk')

if __name__ == "__main__":
    predict_liver("model_liver-epoch_2.kerasmodel")
from keras.models import load_model
import numpy as np
import os
import nibabel as nib
import skimage.io

weights = [0.4, 0.6]

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
    model = load_model(model, custom_objects={'loss': dice_coef_loss_weight, 'metrics': dice_coef})
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
    pair_img = nib.Nifti1Pair(img_preds, np.eye(4))
    nib.save(pair_img, "pred_liver-20.img")
    #compute dice coefficient
    dic = dice_coef_2class(img_preds, segs)
    return dic

def dice_coef_2class(pred, ground):
    n_p = 0
    n_g = 0
    n_pg = 0
    print pred[64,]
    print ground[64,]
    pred = pred.flatten()
    ground = ground.flatten()
    for i in range(0, len(pred)):
        if ground[i] == 1 and pred[i] == 1:
            n_pg += 1
        n_p += pred[i]
        n_g += ground[i]
    return 2*float(n_pg) / float(n_p + n_g)


if __name__ == "__main__":
    #predict_liver("model_liver-epoch_2.kerasmodel")
    dic = predict_liver("model_liver-epoch_2.kerasmodel")
    print dic
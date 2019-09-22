from keras_preprocessing.image import load_img
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def dice_coeff_np(pred, true_im):
    return np.sum(pred[true_im == 1]) * 2.0 / (np.sum(pred) + np.sum(true_im))


def jaccard(im1, im2):
    im1 = np.array(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())


def jaccard_custom(im1, im2):
    intersection = (im1 * im2).sum()
    union = im1.sum() + im2.sum()
    return intersection / (union - intersection)


# -------------

pred_im_path = '../../scripts/map_pred_0.tif'
true_im_path = '../../scripts/train_0_MAP.tif'

pred_im = np.array(load_img(pred_im_path, grayscale=True))
true_im = np.array(load_img(true_im_path, grayscale=True))
pred_im[pred_im == 255] = 1
true_im[true_im == 255] = 1

p = precision_score(true_im.flatten(), pred_im.flatten())
# p1 = (pred_im * true_im).sum() / pred_im.sum()
r = recall_score(true_im.flatten(), pred_im.flatten())
f1 = f1_score(true_im.flatten(), pred_im.flatten())
# jaccard0 = jaccard(true_im, pred_im)
jaccard_ = jaccard_custom(true_im, pred_im)
dice = dice_coeff_np(true_im, pred_im)

print('Precision:', p)
print('Recall:', r)
print('F1_score:', f1)
print('Jaccard_coef:', jaccard_)
print('Dice_coef:', dice)

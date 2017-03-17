import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

img_file = '../../sakaka_data/google_earth_img/test.png'
img = cv2.imread(img_file)

img_file = '../../sakaka_data/google_earth_img/mask.png'
img_mask = cv2.imread(img_file, 0)
ret, mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

# PLOT RAW IMG & MASK
# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
# ax[0].imshow(img), ax[0].set_title('img'), ax[0].axis('off')
# ax[1].imshow(mask.reshape(img.shape[:2])), ax[1].set_title('mask'), ax[1].axis('off')
# plt.tight_layout(), plt.show()

print(img.shape, mask.shape)

img_ = img.reshape(-1, 3)
mask_ = mask.reshape(-1)

# MODEL TRAIN&PREDICT
pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
pipeline.fit(X=img_, y=mask_)
mask_pred = pipeline.predict_proba(img_)[:, 1]

mask_pred_thresh = mask_pred >= np.percentile(mask_pred, 96)

# PLOTS
fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))
ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
ax[1].imshow(mask), ax[1].set_title('mask_true'), ax[1].axis('off')
ax[2].imshow(mask_pred.reshape(mask.shape)), ax[2].set_title('mask_pred'), ax[2].axis('off')
ax[3].imshow(mask_pred_thresh.reshape(mask.shape)), ax[3].set_title('mask_pred_thresh'), ax[3].axis('off')
plt.tight_layout(), plt.show()
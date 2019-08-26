import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

import cv2


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


file_path = os.path.normpath('data/test/geo/Mecca/mecca_zoom19_1-3.tiff')
########################################################

img_sat_ = cv2.imread(file_path, 1)  # [2000:2256, 0:256]


# Convert to grayscale
img_sat_ = cv2.cvtColor(img_sat_, cv2.COLOR_BGR2GRAY)



# img_sat = img_sat_.astype('float32')
# img_sat /= 255

# Brightness and contrast:
img_sat_ = apply_brightness_contrast(img_sat_, -100, 77)

# Exposure
img_sat_ = img_sat_ * 1.5


print('Saving')
cv2.imwrite('out5.tiff', img_sat_)

# Median filtering
gray_image_mf = median_filter(img_sat_, 1)

# Calculate the Laplacian
lap = cv2.Laplacian(gray_image_mf, cv2.CV_64F)

# Calculate the sharpened image
sharp = img_sat_ - 0.7 * lap




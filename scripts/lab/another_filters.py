import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter

import cv2



file_path = os.path.normpath('out5.tiff')
########################################################
mask = cv2.imread(file_path, cv2.COLOR_GRAY2BGR)  # [2000:2256, 0:256]

mask[mask > 250] = 255
mask[mask < 250] = 0


print('Saving')
cv2.imwrite('mask_wihtoutsharp_250.tiff', mask)


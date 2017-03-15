import os
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from area_assesment.data_processing.data_io import *

# img = tiff.imread('22678930_15.tiff')
#
# mask = cv2.imread('22678930_15.tiff')
# print(mask)
#
# plt.imshow(mask), plt.show()



# mask = cv2.imread('22678930_15.tiff')
# print(mask)
#
# plt.imshow(mask), plt.show()

print(filenames_in_dir('.', endswith_='.tiff'))
files = filenames_in_dir('.', endswith_='.tiff')
for f in files:
    img = read_img(f)
    plt.imshow(img), plt.show()
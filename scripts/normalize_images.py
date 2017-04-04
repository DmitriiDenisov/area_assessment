import os
import numpy as np
import cv2
import logging

from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import *

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# dir_valid_sat = os.path.normpath(
#     '/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal_B_1m/')
dir = os.path.normpath('../sakaka_data/nonequal_imgs/')
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir))
files = filenames_in_dir(dir, endswith_='.tif')
output_folder = os.path.normpath('../sakaka_data/buildings/output/')

for i, f_sat in enumerate(files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(files), f_sat))
    img_sat_ = cv2.imread(f_sat)  # [-500:, -1000:]
    img_sat_eq = hisEqulColor(img_sat_.copy())

    plot2(img_sat_, img_sat_eq, name='{}_equalizeHist'.format(os.path.basename(f_sat)[:-4]),
          show_plot=True, save_output_path=None)

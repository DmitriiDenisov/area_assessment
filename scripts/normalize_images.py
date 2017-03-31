import os
import numpy as np
import cv2
import logging

from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])


dir_valid_sat = os.path.normpath(
    '/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal_B_1m/')
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir_valid_sat))
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')
output_folder = os.path.normpath('../sakaka_data/buildings/output/')

for i, f_sat in enumerate(valid_sat_files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))
    img_sat_ = cv2.imread(f_sat)  # [-500:, -1000:]
    img_sat = equalizeHist_rgb(img_sat_)

    plot_img(img_sat, name='{}_equalizeHist'.format(f_sat.split('/')[-1][:-4]),
             show_plot=True, save_output_path=output_folder)
import os
import numpy as np
import cv2
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask


def equalizeHist_rgb(img):
    # return rgb/255.0
    norm = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    norm[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    norm[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    norm[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return norm

if __name__ == '__main__':
    valid_sat_files = os.path.normpath('C:\\Users\\ktsyganov\\Documents\SAKAKA\\sakaka_solution\\area_assessment_repo\\area_assesment\\sakaka_data\\New folder')  # '../../data/mass_buildings/valid/'

    output_folder = os.path.normpath('../sakaka_data/buildings/output/')
    valid_sat_files = filenames_in_dir(valid_sat_files, endswith_='.tif')
    for i, f_sat in enumerate(valid_sat_files):
        print(f_sat)

        img_sat_ = cv2.imread(f_sat)  # [-500:, -1000:]
        img_sat = equalizeHist_rgb(img_sat_)
        plot_img_mask(img_sat_, img_sat, show_plot=True)
import cv2
import os
import numpy as np
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score
from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
from area_assesment.io_operations.visualization import plot2, plot3
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.tf_unet.image_util import ImageDataProvider
from area_assesment.neural_networks.tf_unet.unet import Unet, Trainer
from area_assesment.neural_networks.unet import *
import hashlib

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 images_augmentation.py
#########################################################


logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir_train = os.path.normpath('../sakaka_data/buildings/train2/*.tif')
dir_unet = os.path.normpath('../unet_trained/')

data_provider = ImageDataProvider('../sakaka_data/buildings/train2/*.tif')
net = Unet(layers=3, features_root=64, channels=3, n_class=3)
trainer = Trainer(net)
path = trainer.train(data_provider, dir_unet, training_iters=32, epochs=100)

prediction = net.predict(path, data_provider)
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

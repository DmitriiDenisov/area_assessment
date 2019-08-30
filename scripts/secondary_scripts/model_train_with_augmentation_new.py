import os
import numpy as np
import logging
from keras.callbacks import ModelCheckpoint
from os.path import isfile, join
from os import listdir

from area_assesment.data_processing.DataGeneratorCustom import DataGeneratorCustom
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot2
from area_assesment.neural_networks.cnn import *

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 images_augmentation.py
#########################################################
from area_assesment.neural_networks.logger import TensorBoardBatchLogger
from area_assesment.neural_networks.unet import unet

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])


# MODEL DEFINITION
logging.info('MODEL DEFINITION')

if True:
    model = unet(64, 64, 3)
    model.summary()
    pass
else:
    # LOADING PREVIOUS WEIGHTS OF MODEL
    logging.info('LOADING PREVIOUS WEIGHTS OF MODEL: {}'.format(net_weights_load))
    model.load_weights(net_weights_load)



# PATCHING SETTINGS
nn_input_patch_size = (64, 64) # (1024, 1024)  # (1024, 1024)  # (64, 64)
nn_output_patch_size = (64, 64) # (128, 128)  # (256, 256) # (16, 16)
step_size = 32  # 256  # 16

# MODEL SETTINGS
epochs = 100
net_weights_load = None  # os.path.normpath('../weights/cnn_circlefarms/w_epoch97_jaccard0.6741_valjaccard0.4200.hdf5')
net_weights_dir_save = os.path.normpath('../../weights/unet_mecca')
########################################################

# Generators
train_dir = '../../data/train/sat'
val_dir = '../../data/val/sat'
file_names = filenames_in_dir(train_dir, endswith_='.tif')
train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]
train_data_gen = DataGeneratorCustom(batch_size=4, train_dir='../../data/train/sat', train_masks_dir='../../data/train/map', patch_size=nn_input_patch_size, step_size=step_size)
step_per_epoch = train_data_gen.step_per_epoch
train_data_gen = iter(train_data_gen)
val_data_gen = DataGeneratorCustom(batch_size=4, train_dir='../../data/val/sat', train_masks_dir='../../data/val/map', patch_size=nn_input_patch_size, step_size=step_size)
step_per_val = val_data_gen.step_per_epoch
val_data_gen = iter(val_data_gen)


# FIT MODEL AND SAVE WEIGHTS
logging.info('FIT MODEL, EPOCHS: {}, SAVE WEIGHTS: {}'.format(epochs, net_weights_dir_save))

tb_callback = TensorBoardBatchLogger(project_path='../../', batch_size=1, log_every=1)
checkpoint = ModelCheckpoint(os.path.join(net_weights_dir_save,
                             'w_epoch{epoch:02d}_jaccard{jaccard_coef:.4f}_valjaccard{val_jaccard_coef:.4f}.hdf5'),
                             monitor='val_jaccard_coef', save_best_only=True)

a = 3

model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=step_per_epoch,
    validation_data=val_data_gen,
    validation_steps=step_per_val,
    epochs=epochs,
    callbacks=[checkpoint, tb_callback],
    verbose=1)

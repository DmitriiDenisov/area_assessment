import os, sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать
import logging
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from area_assesment.neural_networks.DataGeneratorCustom import DataGeneratorCustom
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.logger import TensorBoardBatchLogger
from area_assesment.neural_networks.unet import unet, uresnet, unet2, unet_old, CreateModel_uresnet

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.ERROR,
                    handlers=[logging.StreamHandler()])

# SETTINGS
nn_input_patch_size = (64, 64)  # (1024, 1024)  # (1024, 1024)  # (64, 64)
nn_output_patch_size = (64, 64)  # (128, 128)  # (256, 256) # (16, 16)
step_size = 32  # 256  # 16

epochs = 400
batch_size = 4
net_weights_load = None
net_weights_load = '../weights/unet_mecca/new_model_w_epoch80_jaccard0.9150.hdf5'
# net_weights_load = '../weights/unet/unet_adam_64x64_epoch01_jaccard0.9510_valjaccard0.9946.hdf5'
# net_weights_load = '../weights/cnn_v7/sakaka_cnn_v7_jaccard0.2528_valjaccard0.0406.hdf5'
# net_weights_load = '../weights/cnn_v7/w_epoch03_jaccard0.3890_valjaccard0.1482.hdf5'
net_weights_dir_save = os.path.normpath('../weights/unet_mecca')
train_dir = '../data/train/sat'
train_masks_dir = '../data/train/map'
train_nokia_poly = '../data/train/nokia_map'
val_dir = '../data/val/sat'
val_masks_dir = '../data/val/map'
val_nokia_poly = '../data/val/nokia_map'

# MODEL DEFINITION

if not net_weights_load:
    # model = unet_old(64, 64, 4)
    # model.summary()
    model = unet2((64, 64, 3))
    # model = uresnet(input_layer=)
    # model = unet(input_size=(64, 64, 1))
    # input_layer = Input((64, 64, 3))
    # output_layer = uresnet(input_layer, 16, 0.2)
    # model = Model(input_layer, output_layer)
    # model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[dice_coef_K])
    # model = CreateModel_uresnet(img_size_target=64)
    # model.compile(optimizer='adam', loss='binary_crossentropy',
    #              metrics=[precision, recall, fmeasure, dice_coef_K, jaccard_coef])
    model.summary()

else:
    # LOADING PREVIOUS WEIGHTS OF MODEL
    logging.info('LOADING PREVIOUS WEIGHTS OF MODEL: {}'.format(net_weights_load))
    model = load_model(net_weights_load,
                       custom_objects={
                           "precision": precision,
                           "recall": recall,
                           "fmeasure": fmeasure,
                           "jaccard_coef": jaccard_coef,
                           "dice_coef_K": dice_coef_K
                       }
                       )
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[precision, recall, fmeasure, dice_coef_K, jaccard_coef])

target_shape = model.output_shape
# model.load_weights(net_weights_load)


# DATA GENERATORS
file_names = filenames_in_dir(train_dir, endswith_='.tif')
train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]
train_data_gen = DataGeneratorCustom(batch_size=batch_size,
                                     train_dir=train_dir,
                                     train_masks_dir=train_masks_dir,
                                     train_nokia_poly=train_nokia_poly,
                                     patch_size=nn_input_patch_size,
                                     step_size=step_size,
                                     target_shape=target_shape,
                                     nokia_map=False)
step_per_epoch = train_data_gen.step_per_epoch // batch_size
train_data_gen = iter(train_data_gen)
val_data_gen = DataGeneratorCustom(batch_size=batch_size,
                                   train_dir=val_dir,
                                   train_masks_dir=val_masks_dir,
                                   train_nokia_poly=val_nokia_poly,
                                   patch_size=nn_input_patch_size,
                                   step_size=step_size,
                                   target_shape=target_shape,
                                   nokia_map=False)
step_per_val = val_data_gen.step_per_epoch // batch_size
val_data_gen = iter(val_data_gen)

# FIT MODEL AND SAVE WEIGHTS
logging.info('FIT MODEL, EPOCHS: {}, SAVE WEIGHTS: {}'.format(epochs, net_weights_dir_save))

tb_callback = TensorBoardBatchLogger(project_path='../', log_every=4)
checkpoint = ModelCheckpoint(os.path.join(net_weights_dir_save,
                                          'retrained_10sept_w_epoch{epoch:02d}_jaccard{jaccard_coef:.4f}.hdf5'), #valjaccard{val_jaccard_coef:.4f}
                             monitor='jaccard_coef', save_best_only=True)

model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=step_per_epoch,
    # validation_data=val_data_gen,
    # validation_steps=step_per_val,
    epochs=epochs,
    callbacks=[checkpoint, tb_callback],
    verbose=1)

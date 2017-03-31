from keras.engine import Model
from keras.layers import *
from keras.optimizers import SGD

from area_assesment.neural_networks.metrics import jaccard_coef


def unet(patch_height, patch_width, n_ch):
    inputs = Input((patch_height, patch_width, n_ch))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=3)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    # #
    conv6 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(conv5)
    conv6 = core.Reshape((patch_height, patch_width))(conv6)
    # ############
    conv7 = core.Activation('sigmoid')(conv6)

    model = Model(input=inputs, output=conv7)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', jaccard_coef])
    return model

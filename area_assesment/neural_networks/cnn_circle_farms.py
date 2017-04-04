from keras.engine import Model
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import SGD
from area_assesment.neural_networks.metrics import *


def cnn_circle_farms_v1(patch_height, patch_width, n_ch):
    inputs = Input((patch_height, patch_width, n_ch))

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, border_mode='same')(conv1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=1, border_mode='same')(conv2)
    drop1 = Dropout(0.2)(pool2)

    conv3 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(drop1)
    resh1 = core.Reshape((patch_height*patch_width, 2))(conv3)

    activ = core.Activation('softmax')(resh1)
    resh2 = core.Reshape((patch_height, patch_width, 2))(activ)

    model = Model(input=inputs, output=resh2)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy', precision,
                                                                        recall, fmeasure, jaccard_coef])
    return model


def cnn_circlefarms_v2():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=16, strides=4, activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Conv2D(filters=112, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(filters=80, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(64 ** 2, activation='sigmoid'))
    model.add(Reshape((64, 64)))
    # print(model.output_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision,
                                                                         recall, fmeasure, jaccard_coef])
    return model
from keras.models import Sequential
from keras.layers import *


def cnn_v1():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=16, strides=4, activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Conv2D(filters=112, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(filters=80, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(64**2, activation='sigmoid'))
    model.add(Reshape((64, 64)))
    # print(model.output_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model():
    model = Sequential()
    # model.add(Convolution2D(filters=16, kernel_size=4, input_shape=(64, 64, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(64**2, activation='sigmoid'))
    model.add(Reshape((64, 64)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

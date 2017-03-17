from keras.models import Sequential
from keras.layers import *


def cnn_v1():
    """
    Definition of cnn architecture from
    Article: "Multiple Object Extraction from Aerial Imagery with Convolutional Neural Networks" by
    Authors: Saito, Shunta; Yamashita, Takayoshi; Aoki, Yoshimitsu
    Source: http://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003

    The difference is in tha last dense layer:
    1) shape
    2) activation function is sigmoid instead of softmax (due to binary class problem specification)

    :return: keras model
    """

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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model():
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3), activation='relu'))
    model.add(Dense(64**2, activation='sigmoid'))
    model.add(Reshape((64, 64)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

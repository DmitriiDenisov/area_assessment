from keras.models import Sequential
from keras.layers import *


def cnn_v1():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=16, strides=4, input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Conv2D(filters=112, kernel_size=4, strides=1))
    model.add(Conv2D(filters=80, kernel_size=3, strides=1))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(64**2, activation='softmax'))
    model.add(Reshape((64, 64)))
    # print(model.output_shape)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

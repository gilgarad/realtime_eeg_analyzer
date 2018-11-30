from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras import optimizers
import keras.backend.tensorflow_backend as K
import numpy as np


class CNN:
    def __init__(self, input_shape):
        with K.tf.device('/gpu:0'):
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(14, 14), activation='relu',
                             input_shape=input_shape))
            model.add(Conv2D(64, kernel_size=(14, 1)))
            #     model.add(Conv2D(64, kernel_size=(14, 1)))
            #     model.add(Conv2D(64, kernel_size=(14, 1)))
            #     model.add(Conv2D(64, kernel_size=(14, 1)))

            #     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
            #     model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Flatten())
            #     model.add(Dense(32))
            #     model.add(Dropout(rate=0.5))
            model.add(Dense(1))

            #     rmsprop = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08)
            rmsprop = optimizers.RMSprop()
            adam = optimizers.Adam()
            #     sgd = optimizers.SGD()
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        self.model = model

    def test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('CNN Test')
        print('##########')

        _x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        _x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        print(_x_train.shape)
        print(_x_test.shape)

        self.model.fit(_x_train, y_train, validation_data=(_x_test, y_test), epochs=100, batch_size=32)
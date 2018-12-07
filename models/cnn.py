from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Activation, Input
from keras import optimizers
import keras.backend.tensorflow_backend as K
import numpy as np
from keras.utils import np_utils
from keras.layers.merge import add


class CNN:
    def __init__(self, input_shape):
        with K.tf.device('/gpu:0'):
            inputs = Input(input_shape)
            L = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=inputs.get_shape().as_list(),
                       padding='same')(inputs)

            for i in range(10):
                L = self.resnet(L)

            flatten = Flatten()(L)
            output = Dense(1, input_shape=flatten.get_shape())(flatten)

            model = Model(inputs=inputs, outputs=output)
            #     rmsprop = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08)
            rmsprop = optimizers.RMSprop()
            adam = optimizers.Adam()
            #     sgd = optimizers.SGD()
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        model.summary()
        self.model = model

    def resnet(self, layer):
        L = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=layer.get_shape().as_list(), padding='same')(
            layer)
        L = Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=L.get_shape().as_list(), padding='same')(L)
        L = add([L, layer])
        L = Activation('relu')(L)
        return L

    def test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('CNN Test')
        print('##########')

        _x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        _x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        print(_x_train.shape)
        print(_x_test.shape)

        self.model.fit(_x_train, y_train, validation_data=(_x_test, y_test), epochs=30, batch_size=4)

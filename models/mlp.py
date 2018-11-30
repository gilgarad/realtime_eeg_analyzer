from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras import optimizers
import keras.backend.tensorflow_backend as K


class MLP:
    def __init__(self, input_shape):
        with K.tf.device('/gpu:0'):
            model = Sequential()

            model.add(Dense(64, activation='sigmoid'))
            model.add(Dense(64, activation='sigmoid'))
            if len(input_shape) == 3:
                model.add(Flatten())
            model.add(Dense(64, activation='sigmoid'))
            model.add(Dense(1, activation='sigmoid'))

            adam = optimizers.Adam()
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        self.model = model

    def test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('MLP Test')
        print('##########')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

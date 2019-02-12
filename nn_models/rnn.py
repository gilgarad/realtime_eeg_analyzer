from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM, BatchNormalization, Activation, Dropout
from keras.layers.wrappers import Bidirectional
from keras import optimizers
import keras.backend.tensorflow_backend as K


class RNN:
    def __init__(self, input_shape, gpu=0):
        with K.tf.device('/gpu:' + str(gpu)):
            model = Sequential()
            model.add(Bidirectional(LSTM(128, return_sequences=False)))
            model.add(BatchNormalization())
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            model.add(Dense(3))

            adam = optimizers.Adam()
            #     rmsprop = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08)
            #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        self.model = model

    def test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('RNN Test')
        print('##########')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM
from keras import optimizers
import keras.backend.tensorflow_backend as K


class RNN:
    def __init__(self, input_shape):
        with K.tf.device('/gpu:0'):
            model = Sequential()
            model.add(LSTM(128, return_sequences=True))
            #     model.add(LSTM(128, return_sequences=True))
            #     model.add(Dropout(rate=0.5))
            model.add(LSTM(128, return_sequences=False))
            #     model.add(Dense(1, activation='relu'))
            model.add(Dense(1))
            #     model.add(TimeDistributed(Dense(1)))

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


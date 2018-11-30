from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras import optimizers
import keras.backend.tensorflow_backend as K
from tcn import TCN as TCN_original


class TCN:
    def __init__(self, input_shape):
        with K.tf.device('/gpu:0'):
            i = Input(batch_shape=input_shape)

            o = TCN_original(return_sequences=False)(i)  # The TCN layers are here.
            o = Dense(1, activation='sigmoid')(o)

            model = Model(inputs=[i], outputs=[o])
            adam = optimizers.Adam()
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        self.model = model

    def test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('TCN Test')
        print('##########')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from keras import optimizers
import keras.backend.tensorflow_backend as K
from sklearn.metrics import classification_report
import numpy as np


class MLP:
    def __init__(self, input_shape, num_layers=3, num_classes=3):
        with K.tf.device('/gpu:0'):
            model = Sequential()

            for i in range(num_layers):
                model.add(Dense(1024, activation=None))
                model.add(BatchNormalization())
                model.add(Activation('tanh'))
                model.add(Dropout(0.5))

            if len(input_shape) == 3:
                print()
                model.add(Flatten())
            model.add(Dense(num_classes, activation=None))
            model.add(BatchNormalization())
            model.add(Activation('softmax'))

            adam = optimizers.Adam()
            sgd = optimizers.SGD()
            #             model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model = model

    def test(self, x_train, y_train, x_test, y_test, verbose=1):
        print('##########')
        print('MLP Test')
        print('##########')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        # self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=64,
        #                callbacks=['early_stopping'])
        self.model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=64,
                       verbose=verbose, callbacks=[early_stopping], shuffle=True)
        # loss, metrics = self.model.evaluate(x=x_test, y=y_test, batch_size=64)
        # print(metrics)
        _y = self.model.predict(x=x_test, batch_size=128)
        _y = np.argmax(_y, axis=1)
        y_test = np.argmax(y_test, axis=1)

        accuracy = [y_test == _y][0]
        accuracy = float(len(accuracy[accuracy==True]) / len(y_test))
        print('##########')
        print("Accuracy: %.4f" % (accuracy))
        print('##########')
        print(classification_report(y_test, _y))

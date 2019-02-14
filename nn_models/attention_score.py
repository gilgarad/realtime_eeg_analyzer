from keras.layers import Multiply, Dense, Permute, Lambda, Input, Dropout, Reshape, Flatten
# from keras.layers.core import *
from keras.models import Model
from keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras import optimizers


# Define early stop and learning rate decay !!!
class LrReducer(Callback):
    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=3, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = -1
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
        self.initial_learning_rate = -1

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')
        if self.best_loss == -1 and self.initial_learning_rate == -1:
            self.best_loss = current_loss
            self.initial_learning_rate = K.get_value(self.model.optimizer.lr)

        if current_loss <= self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.verbose > 2:
                print('---current best loss: %.3f' % current_loss)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr, lr * self.reduce_rate)
                    self.wait = -1
                    if self.verbose > 0:
                        print("Learning Rate Decay: %.5f -> %.5f" % (lr, lr * self.reduce_rate))
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch + 1))
                    self.model.stop_training = True

            if K.get_value(self.model.optimizer.lr) * 100 < self.initial_learning_rate:
                if self.verbose > 0:
                    print("Epoch %d: early stopping" % (epoch + 1))
                self.model.stop_training = True
            self.wait += 1


def ScoreActivationFromSigmoid(x, target_min=1, target_max=9):
    activated_x = K.sigmoid(x)
    return activated_x * (target_max - target_min) + target_min


def GetPadMask(q):
    mask = K.cast(K.not_equal(K.sum(q, axis=-1, keepdims=True), 0), 'float32')
    return mask


def GetCountNonZero(x):
    return 1 / tf.reduce_sum(tf.cast(x, 'float32'), axis=-2, keepdims=True)


# Best Working
class AttentionScores:
    def __init__(self, input_shape, output_labels, gpu=0):
        with K.tf.device('/gpu:' + str(gpu)):
            inputs = Input(shape=input_shape)

            num_attentions = 64
            num_neurons = 128

            m = inputs
            mask = Lambda(lambda x: GetPadMask(x))(m)
            nonzero_mask_count = Lambda(lambda x: GetCountNonZero(x))(mask)

            m = Dense(num_neurons, activation='relu')(m)
            m = Dropout(0.7)(m)

            all_final_scores = list()
            for label_score in output_labels:
                _m = Dense(num_attentions, activation=ScoreActivationFromSigmoid)(m)  # n attentions
                _m = Multiply()([mask, _m])
                _m = Multiply()([_m, nonzero_mask_count])

                _m = Permute((2, 1))(_m)

                _m = Lambda(lambda x: K.sum(x, axis=2))(_m)  # each attention's average

                _m = Reshape((num_attentions, 1))(_m)
                final_score = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(
                    _m)  # average of all attention's averages
                #                 final_score = Multiply(name=label_score)([final_score, Lambda(lambda x: K.ones_like(x))(final_score)])
                final_score = Flatten(name=label_score)(final_score)
                all_final_scores.append(final_score)

            model = Model(inputs=inputs, outputs=all_final_scores)

        losses = {'amusement': 'mean_squared_error',
                  'immersion': 'mean_squared_error',
                  'difficulty': 'mean_squared_error',
                  'emotion': 'mean_squared_error'}
        loss_weights = {'amusement': 1.0, 'immersion': 1.0, 'difficulty': 1.0, 'emotion': 1.0}
        adam = optimizers.Adam(lr=0.001)
        model.compile(loss=losses, loss_weights=loss_weights, optimizer=adam, metrics=['accuracy'])

        self.model = model
        print(model.summary())

    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128, verbose=1):
        print('##########')
        print('MLP Attention Test')
        print('##########')
        #         early_stopping = EarlyStopping(patience = 20)
        early_stopping = LrReducer(patience=5, reduce_rate=0.5, reduce_nb=3, verbose=1)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                       verbose=verbose,
                       callbacks=[early_stopping])
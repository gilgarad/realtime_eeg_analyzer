from keras.callbacks import EarlyStopping, Callback
import keras.backend.tensorflow_backend as K


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

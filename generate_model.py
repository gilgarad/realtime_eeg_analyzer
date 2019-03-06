# Train and save model
# load and test model

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K

from sklearn.model_selection import KFold
import argparse

from neural_network.nn_models.attention_score import AttentionScoreFourierTransform, AttentionScore

from neural_network.utils.dataset import Dataset


class ModelRunner:
    def __init__(self, data_path):
        self.sess = self._session_init()
        self.data_path = data_path
        self.dataset = self._load_dataset(self.data_path)
        self.model_dict = {'attention_score': AttentionScore,
                           'attention_score_ft': AttentionScoreFourierTransform}

    def add_model(self, model_name, model):
        self.model_dict[model_name] = model

    def _session_init(self):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        K.set_session(sess)
        return sess

    def _load_dataset(self, data_path):
        return Dataset(data_path=data_path, num_channels=14, max_minutes=10, num_original_features=18,
                       num_reduced_features=10)

    def train(self, model_name, multiloss=False, gpu=0, epochs=30, batch_size=100, verbose=1):
        model = self.model_dict[model_name]
        data_list = list(self.dataset.data_dict.keys())
        print('Number of Trials: %i' % len(data_list))
        print('Trial Names: %s' % data_list)
        print('')

        data_portion_idx = 7

        data_list = np.array(data_list)
        kf = KFold(n_splits=10, shuffle=True, random_state=1518) # 10 percent
        data_list2 = kf.split(data_list)
        data_list2 = list(data_list2)
        train_names = data_list[data_list2[data_portion_idx][0]]
        validation_names = data_list[data_list2[data_portion_idx][1]]

        x_train, y_train, x_test, y_test = self.dataset.get_data(data_dict=self.dataset.data_dict,
                                                                 train_names=train_names, test_names=validation_names,
                                                                 feature_type='rawdata', is_classification=False)

        train_model = None
        if multiloss:
            train_model = self.train_multiloss
        else:
            train_model = self.train_singleloss

        trained_model = train_model(model=model, data=[x_train, y_train, x_test, y_test], gpu=gpu, epochs=epochs,
                                    batch_size=batch_size, verbose=verbose)

        return trained_model

    def train_multiloss(self, model, data, gpu, epochs=150, batch_size=1028, verbose=1):
        x_train, y_train, x_test, y_test = data
        num_classes = len(np.unique(y_train, axis=0))
        _model = model(input_shape=(x_train.shape[1], x_train.shape[2]),
                       output_labels=['amusement',
                                      'immersion',
                                      'difficulty',
                                      'emotion'
                                      ], gpu=gpu)
        _model.train(x_train, {
            'amusement': y_train[0],
            'immersion': y_train[1],
            'difficulty': y_train[2],
            'emotion': y_train[3]
        }, x_test, {
                         'amusement': y_test[0],
                         'immersion': y_test[1],
                         'difficulty': y_test[2],
                         'emotion': y_test[3]
                     }, batch_size=batch_size, epochs=epochs, verbose=verbose)
        keras_model = _model.model

        return keras_model

    def train_singleloss(self, model, data, gpu, epochs=150, batch_size=1028, verbose=1):
        x_train, y_train, x_test, y_test = data
        num_classes = len(np.unique(y_train, axis=0))
        _model = model(input_shape=(x_train.shape[1], x_train.shape[2]), gpu=gpu)
        _model.train(x_train, y_train[0], x_test, y_test[0], batch_size=batch_size, epochs=epochs, verbose=verbose)
        keras_model = _model.model

        return keras_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path

    print('')
    model_runner = ModelRunner(data_path=data_path)
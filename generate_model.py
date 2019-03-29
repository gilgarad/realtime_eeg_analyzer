""" generate_model.py: Train and test model, save and load as well. """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

# Train and save model
# load and test model

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.models import model_from_json

from sklearn.model_selection import KFold
import argparse
import json

from neural_network.nn_models.attention_score import AttentionScoreFourierTransform, AttentionScore
from neural_network.utils.dataset import Dataset
from neural_network.utils.custom_function import get_custom_function_dict


class ModelRunner:
    def __init__(self, data_path, num_channels=14, max_minutes=10, num_original_features=18, num_reduced_features=10,
                 augment=False, stride=128, delete_range=128, data_status='rawdata', sampling_rate=128, padding='all'):
        """ Initialize object for load dataset and train / test / save / load

        :param data_path:
        :param num_channels:
        :param max_minutes:
        :param num_original_features:
        :param num_reduced_features:
        :param augment:
        :param stride:
        :param delete_range:
        :param data_status:
        """
        self.sess = self._session_init()
        self.data_path = data_path
        self.num_channels = num_channels # number of channels
        self.max_minutes = max_minutes
        self.num_original_features = num_original_features
        self.num_reduced_features = num_reduced_features
        self.augment = augment
        self.stride = stride
        self.delete_range = delete_range
        self.data_status = data_status # 'rawdata', 'fourier_transform', 'pre_fourier_transformed'
        self.sampling_rate = sampling_rate
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.padding = padding

        self.dataset = self._load_dataset(self.data_path, num_channels=self.num_channels, max_minutes=self.max_minutes,
                                          num_original_features=self.num_original_features,
                                          num_reduced_features=self.num_reduced_features,
                                          augment=self.augment, stride=self.stride, delete_range=self.delete_range,
                                          data_status=self.data_status, sampling_rate=self.sampling_rate,
                                          padding=self.padding)
        self.model_dict = {'attention_score': AttentionScore,
                           'attention_score_ft': AttentionScoreFourierTransform}

        self.trained_model = None
        self.is_classification = False

    def add_model(self, model_name, model):
        """ Add newly written(coded) model

        :param model_name:
        :param model:
        :return:
        """
        self.model_dict[model_name] = model

    def _session_init(self):
        """ Initialize keras session

        :return:
        """
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        K.set_session(sess)
        return sess

    def _load_dataset(self, data_path, num_channels=14, max_minutes=10, num_original_features=18,
                       num_reduced_features=10, augment=False, stride=128, delete_range=128, data_status='rawdata',
                      sampling_rate=128, padding='all'):
        """ Load dataset for training or testing

        :param data_path:
        :param num_channels:
        :param max_minutes:
        :param num_original_features:
        :param num_reduced_features:
        :param augment:
        :param stride:
        :param delete_range:
        :param data_status:
        :return:
        """
        return Dataset(data_path=data_path, num_channels=num_channels, max_minutes=max_minutes,
                       num_original_features=num_original_features, num_reduced_features=num_reduced_features,
                       augment=augment, stride=stride, delete_range=delete_range, data_status=data_status,
                       sampling_rate=sampling_rate, padding=padding)

    def prepare_dataset(self, is_classification=False):
        """ Process dataset to form correctly for feeding to models

        :param is_classification:
        :return:
        """

        self.is_classification = is_classification

        data_list = list(self.dataset.data_dict.keys())
        print('Number of Trials: %i' % len(data_list))
        print('Trial Names: %s' % data_list)
        print('')

        data_portion_idx = 7

        data_list = np.array(data_list)
        kf = KFold(n_splits=10, shuffle=True, random_state=1518)  # 10 percent
        data_list2 = kf.split(data_list)
        data_list2 = list(data_list2)
        train_names = data_list[data_list2[data_portion_idx][0]]
        validation_names = data_list[data_list2[data_portion_idx][1]]

        self.x_train, self.y_train, self.x_valid, self.y_valid = self.dataset.get_data(data_dict=self.dataset.data_dict,
                                                                                       train_names=train_names,
                                                                                       test_names=validation_names,
                                                                                       feature_type=self.data_status,
                                                                                       is_classification=is_classification)

    def train(self, model_name, multiloss=False, prepare_dataset=True, gpu=0, epochs=30, batch_size=100, verbose=1):
        """ Train model

        :param model_name:
        :param multiloss:
        :param prepare_dataset:
        :param gpu:
        :param epochs:
        :param batch_size:
        :param verbose:
        :return:
        """
        model = self.model_dict[model_name]
        if prepare_dataset:
            self.prepare_dataset()

        if multiloss:
            train_model = self.train_multiloss
        else:
            train_model = self.train_singleloss

        trained_model = train_model(model=model, data=[self.x_train, self.y_train, self.x_valid, self.y_valid],
                                    gpu=gpu, epochs=epochs, batch_size=batch_size, verbose=verbose)

        self.trained_model = trained_model

        return trained_model

    def train_singleloss(self, model, data, gpu=0, epochs=150, batch_size=1028, verbose=1):
        """ Train singleloss type model

        :param model:
        :param data:
        :param gpu:
        :param epochs:
        :param batch_size:
        :param verbose:
        :return:
        """

        x_train, y_train, x_test, y_test = data
        initial_params = model.get_initial_params(x_train=x_train, y_train=y_train[0])
        _model = model(initial_params=initial_params, gpu=gpu)
        _model.train(x_train, y_train[0], x_test, y_test[0], batch_size=batch_size, epochs=epochs, verbose=verbose)
        trained_model = _model.model

        self.trained_model = trained_model

        return trained_model

    def train_multiloss(self, model, data, gpu, epochs=150, batch_size=1028, verbose=1):
        """ Train multiloss type model

        :param model:
        :param data:
        :param gpu:
        :param epochs:
        :param batch_size:
        :param verbose:
        :return:
        """
        x_train, y_train, x_test, y_test = data
        initial_params = model.get_initial_params(x_train=x_train, y_train=y_train)
        _model = model(initial_params=initial_params, gpu=gpu)
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

    # def train_multiloss(self, model, data, gpu, epochs=150, batch_size=1028, verbose=1):
    #     x_train, y_train, x_test, y_test = data
    #     num_classes = len(np.unique(y_train, axis=0))
    #     _model = model(input_shape=(x_train.shape[1], x_train.shape[2]),
    #                    output_labels=['amusement',
    #                                   'immersion',
    #                                   'difficulty',
    #                                   'emotion'
    #                                   ], gpu=gpu)
    #     _model.train(x_train, {
    #         'amusement': y_train[0],
    #         'immersion': y_train[1],
    #         'difficulty': y_train[2],
    #         'emotion': y_train[3]
    #     }, x_test, {
    #         'amusement': y_test[0],
    #         'immersion': y_test[1],
    #         'difficulty': y_test[2],
    #         'emotion': y_test[3]
    #     }, batch_size=batch_size, epochs=epochs, verbose=verbose)
    #     keras_model = _model.model
    #
    #     return keras_model

    def save_trained_model(self, path, model=None):
        """

        :param path:
        :param model:
        :return:
        """
        if model is None:
            model = self.trained_model
        model_json = model.to_json()

        # Save model
        with open(path + '.json', 'w') as f:
            f.write(model_json)

        # Save weights
        model.save_weights(path + '_weights.h5')

        # Save accessories
        model_accessory = dict()
        model_accessory['loss'] = model.loss
        model_accessory['loss_weights'] = model.loss_weights
        model_accessory['is_classification'] = self.is_classification

        with open(path + '_accessory.json', 'w') as f:
            json.dump(model_accessory, f)

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        with open(path + '.json', 'r') as f:
            loaded_model_json = f.read()
            loaded_model = model_from_json(loaded_model_json,
                                           custom_objects=get_custom_function_dict())
        loaded_model.load_weights(path + "_weights.h5")

        with open(path + '_accessory.json', 'r') as f:
            model_accessory = json.load(f)

        loaded_model.compile(loss=model_accessory['loss'], loss_weights=model_accessory['loss_weights'],
                             optimizer='adam', metrics=['accuracy'])

        self.is_classification = model_accessory['is_classification']

        return loaded_model

    def test(self, data_path, model=None):
        if model is None:
            model = self.trained_model

        test_dataset = self._load_dataset(self.data_path, num_channels=self.num_channels, max_minutes=self.max_minutes,
                                          num_original_features=self.num_original_features,
                                          num_reduced_features=self.num_reduced_features,
                                          augment=self.augment, stride=self.stride, delete_range=self.delete_range,
                                          data_status=self.data_status, padding=self.padding)

        x_train, y_train, x_valid, y_valid = test_dataset.get_data(data_dict=test_dataset.data_dict,
                                                                   train_names=list(),
                                                                   test_names=test_dataset.data_dict.keys(),
                                                                   feature_type=self.data_status,
                                                                   is_classification=self.is_classification)

        y_pred = model.predict(x_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path

    print('')
    model_runner = ModelRunner(data_path=data_path)
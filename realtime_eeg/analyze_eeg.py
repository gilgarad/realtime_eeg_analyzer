from os.path import join
import numpy as np

from keras.models import model_from_json
import keras.backend.tensorflow_backend as K

from nn_models.fft_convention import FFTConvention
from utils.similarity import Similarity


class AnalyzeEEG:
    def __init__(self, path):
        self.path = path
        self.fft_conv = None
        self.models = list()

    def load_models(self, model_names):
        # Arousal Valence
        self.fft_conv = FFTConvention(path=self.path)

        # Load Saved Model
        for model, weight in model_names:
            with open(join(self.path, model), 'r') as f:
                loaded_model_json = f.read()
            with K.tf.device('/cpu:0'):
                loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(join(self.path, weight))

            losses = {'amusement': 'categorical_crossentropy',
                      'immersion': 'categorical_crossentropy',
                      'difficulty': 'categorical_crossentropy',
                      'emotion': 'categorical_crossentropy'}
            loss_weights = {'amusement': 1.0, 'immersion': 1.0, 'difficulty': 1.0, 'emotion': 1.0}
            loaded_model.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['accuracy'])
            self.models.append(loaded_model)
        # print('Model Object at initial', self.model)

    def analyze_eeg_data(self, all_channel_data):
        """
        Process all data from EEG data to predict emotion class.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model. And send it to web ap
        """

        # Get feature from EEG data
        feature = self.fft_conv.get_feature(all_channel_data)
        # only ten features retrieved from frequency form
        feature_basic = feature.reshape((14, 18))
        feature_basic = feature_basic[:, :10]
        feature_basic = feature_basic.ravel()

        # Emotion Prediction by Nadzeri's source
        class_ar = Similarity.compute_similarity(feature=feature_basic, all_features=self.fft_conv.train_arousal,
                                                 label_all=self.fft_conv.class_arousal[0])
        class_va = Similarity.compute_similarity(feature=feature_basic, all_features=self.fft_conv.train_valence,
                                                 label_all=self.fft_conv.class_valence[0])
        emotion_class = self.fft_conv.determine_emotion_class(class_ar, class_va)

        x_test = feature_basic.reshape(1, 14, 10, 1)
        # print(x_test.shape)
        # print(x_test.tolist())
        # print('Model Object', self.model)
        ratio = [0.5, 0.5]
        y_pred = None
        for idx, model in enumerate(self.models):
            _y_pred = model.predict(x=x_test, batch_size=1)

            if y_pred is None:
                new_pred = list()
                for y_elem in _y_pred:
                    new_pred.append(ratio[idx] * y_elem)
                y_pred = new_pred
            else:
                # sum
                new_pred = list()
                for y_elem, _y_elem in zip(y_pred, _y_pred):

                    y_elem_sum = np.sum([y_elem, ratio[idx] * _y_elem], axis=0)
                    new_pred.append(y_elem_sum)
                y_pred = new_pred
        # y_pred = self.model.predict(x=x_test, batch_size=1)

        # Fun Prediction
        fun = np.argmax(y_pred[0], axis=1)[0]
        # fun = random.randint(0, 2)

        # Difficulty Prediction
        difficulty = np.argmax(y_pred[1], axis=1)[0]
        # difficulty = random.randint(0, 1)

        # Immersion Prediction
        immersion = np.argmax(y_pred[2], axis=1)[0]
        # immersion = random.randint(0, 1)

        # Emotion Prediction
        emotion = np.argmax(y_pred[3], axis=1)[0]
        # emotion = random.randint(0, 2)

        return emotion_class, class_ar, class_va, fun, difficulty, immersion, emotion

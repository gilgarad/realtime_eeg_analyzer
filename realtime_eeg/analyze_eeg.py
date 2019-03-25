""" analyze_eeg.py: Core module that analyzes the eeg with time steps """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

from os.path import join
import numpy as np

from keras.models import model_from_json, Model
import keras.backend.tensorflow_backend as K

from neural_network.nn_models.fft_convention import FFTConvention
from neural_network.utils.custom_function import ScoreActivationFromSigmoid, GetCountNonZero, GetPadMask
from utils.similarity import Similarity
from datetime import datetime
from collections import Counter

from typing import Type

class AnalyzeEEG:
    def __init__(self, path: str):
        """ Initialize the core module that analyzes EEG

        :param path:
        """
        self.path = path
        self.fft_conv = None
        self.models = list()

        self.num_frame_check = 16
        self.sampling_rate = 128
        self.realtime_eeg_in_second = 3  # Realtime each ... seconds
        self.number_of_channel = 14
        self.count = 0

        self.eeg_seq_length = self.sampling_rate * self.realtime_eeg_in_second
        self.max_seq_length = self.sampling_rate * 60 * 5 #
        self.num_of_average = int(self.sampling_rate / self.num_frame_check)
        self.arousal_all = [2.0] * self.num_of_average
        self.valence_all = [2.0] * self.num_of_average
        self.fun_status = [0] * self.num_of_average
        self.immersion_status = [0] * self.num_of_average
        self.difficulty_status = [0] * self.num_of_average
        self.emotion_status = [0] * self.num_of_average

        self.eeg_realtime = np.zeros((self.number_of_channel, self.max_seq_length), dtype=np.float)
        # self.eeg_realtime = self.eeg_realtime.T.tolist()
        # print('Length:', len(self.eeg_realtime))
        self.fft_seq_data = np.zeros((1, 300, 140), dtype=np.float)
        self.final_score_pred = np.zeros((4, 1))
        self.seq_pos = 0

        self.fun_accum = 0
        self.immersion_accum = 0
        self.difficulty_accum = 0
        self.emotion_accum = 0

        self.fun_records = list()
        self.immersion_records = list()
        self.difficulty_records = list()
        self.emotion_records = list()

        self.record_start_time = 0
        self.record_duration = 0

        self.record_status = False

        # Status dictionary (will be removed once the regression model is fully applied)
        self.fun_stat_dict = { 0: '일반', 1: '재미있음' }
        self.immersion_stat_dict = { 0: '일반', 1: '몰입됨' }
        self.difficulty_stat_dict = { 0: '쉬움', 1: '어려움' }
        self.emotion_stat_dict = { 0: '즐거움', 1: '일반', 2: '짜증' }

    def load_models(self, model_names: str):
        """ Load models that will be used in analysis process

        :param model_names:
        :return:
        """
        # TODO
        # currently 2 models for short term prediction, one for final prediction for the given model_names

        # Arousal Valence
        self.fft_conv = FFTConvention(path=self.path)

        # Load Saved Model
        for idx, (model, weight) in enumerate(model_names):
            with open(join(self.path, model), 'r') as f:
                loaded_model_json = f.read()

            if idx != 2:
                custom_objects = {}

                losses = {'amusement': 'categorical_crossentropy',
                          'immersion': 'categorical_crossentropy',
                          'difficulty': 'categorical_crossentropy',
                          'emotion': 'categorical_crossentropy'}

            else:
                custom_objects = {
                    'ScoreActivationFromSigmoid': ScoreActivationFromSigmoid,
                    'GetPadMask': GetPadMask, 'GetCountNonZero': GetCountNonZero}

                losses = {'amusement': 'mean_squared_error',
                          'immersion': 'mean_squared_error',
                          'difficulty': 'mean_squared_error',
                          'emotion': 'mean_squared_error'}

            with K.tf.device('/cpu:0'):
                loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
            loaded_model.load_weights(join(self.path, weight))
            loss_weights = {'amusement': 1.0, 'immersion': 1.0, 'difficulty': 1.0, 'emotion': 1.0}
            loaded_model.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['accuracy'])
            self.models.append(loaded_model)

        # print('Model Object at initial', self.model)

    def set_record_status(self, analyze_status: int):
        """ By the external command (like UI), set the record status to start and or stop record the EEG
        and its analysis data

        :param analyze_status:
        :return:
        """
        if analyze_status == 1 and not self.record_status:
            self.record_start_time = datetime.now()
            self.record_status = True
        elif analyze_status == 2 and self.record_status:
            self.record_status = False
            self.analyze_final_prediction()
        elif analyze_status == 3: # reset all recorded data
            self.emotion_records = list()
            self.fun_records = list()
            self.difficulty_records = list()
            self.immersion_records = list()
            self.record_start_time = 0
            self.record_duration = 0
            self.fft_seq_data = np.zeros((1, 300, 140), dtype=np.float)
            self.seq_pos = 0
            self.final_score_pred = np.zeros((4, 1))

    def store_eeg_rawdata(self, eeg_rawdata: list):
        """ Stores the new rawdata to the matrix that holds the data a certain period of time.

        :param eeg_rawdata:
        :return:
        """

        new_data = eeg_rawdata[3: 3 + self.number_of_channel]
        self.eeg_realtime = np.insert(self.eeg_realtime, self.max_seq_length, new_data, axis=1)
        self.eeg_realtime = np.delete(self.eeg_realtime, 0, axis=1)
        self.count += 1

    def build_middle_layer_pred(self, model: Type[Model]):
        """ Function to get the middle-layer prediction of the model

        :param model:
        :return:
        """
        show_layers = [13, 14, 15, 16, 33, 34, 35, 36] # 0 ~ 36 (13~16step scores, 33~34 final scores)
        layers_outputs = list()
        for i in range(1, 37):
            if i in show_layers:
                layers_outputs.append(model.layers[i].output)

        middle_layer_output = K.function([model.layers[0].input],
                                         layers_outputs)

        return middle_layer_output

    def analyze_eeg_data(self, all_channel_data: np.ndarray):
        """ Process all data from EEG data to predict.
        Currently four labels (Amusement, Immersion, Difficulty, Emotion). NOT USED FOR NOW

        :param all_channel_data:
        :return:
        """
        """ 
        
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

        ratio = [0.5, 0.5, 0.5]
        y_pred = None
        for idx, model in enumerate(self.models):
            if idx == 2:
                # print('Middle Layer pred', datetime.now())
                _x_test = feature_basic.reshape(1, 1, 140)
                fft_seq_data = np.zeros((1, 300, 140), dtype=np.float)
                fft_seq_data = np.insert(fft_seq_data, 0, _x_test, axis=1)
                fft_seq_data = np.delete(fft_seq_data, 300, axis=1)
                _y_pred = model.predict([fft_seq_data])
                # print(idx, _y_pred)
                # break
            else:
                continue
        #         _y_pred = model.predict(x=x_test, batch_size=1)
        #         print(idx, _y_pred)
        #
        #     if y_pred is None: # first
        #         y_pred = list()
        #         for y_elem in _y_pred:
        #             y_pred.append(ratio[idx] * y_elem)
        #     else: # sum
        #         new_pred = list()
        #         for y_elem, _y_elem in zip(y_pred, _y_pred):
        #             y_elem_sum = np.sum([y_elem, ratio[idx] * _y_elem], axis=0)
        #             new_pred.append(y_elem_sum)
        #         y_pred = new_pred
        #
        # fun = np.argmax(y_pred[0], axis=1)[0] # Fun Prediction
        # difficulty = np.argmax(y_pred[1], axis=1)[0] # Difficulty Prediction
        # immersion = np.argmax(y_pred[2], axis=1)[0] # Immersion Prediction
        # emotion = np.argmax(y_pred[3], axis=1)[0] # Emotion Prediction

        fun = _y_pred[0][0][0]
        difficulty = _y_pred[1][0][0]
        immersion = _y_pred[2][0][0]
        emotion = _y_pred[3][0][0]
        # print(fun, difficulty, immersion, emotion)
        # print('')

        return emotion_class, class_ar, class_va, fun, difficulty, immersion, emotion, feature_basic

    def analyze_final_prediction(self):
        """ Make a final prediction for entire data in sequence from record start to record end

        :return:
        """
        model = self.models[2] # [TEMP]
        # layers_outputs = list()
        # for i in range(1, 38):
        #     layers_outputs.append(model.layers[i].output)
        #
        # get_3rd_layer_output = K.function([model.layers[0].input],
        #                                   layers_outputs)
        #
        # _layer_output = get_3rd_layer_output([eeg_data])
        # for i in range(0, 37): # 0 ~ 36 (13~16step scores, 33~34 final scores)
        #     layer_output = _layer_output[i]
        #     print(layer_output)
        #     print(len(layer_output[0]))
        #     print('')

        _y_pred = model.predict([self.fft_seq_data])
        # print('analyze_final_prediction:', _y_pred)
        # print(len(np.where(np.sum(self.fft_seq_data, axis=2) ==0)[1]))
        self.final_score_pred = _y_pred
        return _y_pred

    def analyze_and_evaluate_moment(self):
        """ Analyze and predict the moment of three seconds window size.

        :return:
        """
        # eeg_realtime = np.array(self.eeg_realtime).T
        eeg_realtime = self.eeg_realtime
        # Analyze
        emotion_class, class_ar, class_va, fun, difficulty, immersion, emotion, feature_basic = \
            self.analyze_eeg_data(eeg_realtime[:, -self.eeg_seq_length:])

        # stat_code
        if fun >= 6:
            fun_stat_code = 1
        else:
            fun_stat_code = 0

        if immersion >= 5:
            immersion_stat_code = 1
        else:
            immersion_stat_code = 0

        if difficulty >= 5:
            difficulty_stat_code = 1
        else:
            difficulty_stat_code = 0

        if emotion > 6:
            emotion_stat_code = 0
        elif emotion >= 4:
            emotion_stat_code = 1
        else:
            emotion_stat_code = 2

        # print(fun_stat_code, immersion_stat_code, difficulty_stat_code, emotion_stat_code)

        # Last calculation for moment analysis
        if self.count == self.sampling_rate:
            # print('Sampling Rate:', self.count)
            emotion_dict = {
                1: "fear - nervous - stress - tense - upset",
                2: "happy - alert - excited - elated",
                3: "relax - calm - serene - contented",
                4: "sad - depressed - lethargic - fatigue",
                5: "neutral"
            }
            class_ar = np.round(np.mean(self.arousal_all))
            class_va = np.round(np.mean(self.valence_all))

            # emotion_class = self.fft_conv.determine_emotion_class(class_ar, class_va)

            if self.record_status:
                self.fun_records.append(fun_stat_code)
                self.immersion_records.append(immersion_stat_code)
                self.difficulty_records.append(difficulty_stat_code)
                self.emotion_records.append(emotion_stat_code)
                self.record_duration = (datetime.now() - self.record_start_time).seconds
                x_test = feature_basic.reshape(1, 1, 140)
                if self.seq_pos == 300:
                    self.fft_seq_data = np.insert(self.fft_seq_data, self.seq_pos, x_test, axis=1)
                    self.fft_seq_data = np.delete(self.fft_seq_data, 0, axis=1)
                else:
                    self.fft_seq_data = np.insert(self.fft_seq_data, self.seq_pos, x_test, axis=1)
                    self.fft_seq_data = np.delete(self.fft_seq_data, 300, axis=1)
                    self.seq_pos += 1

            self.count = 0

        if len(self.valence_all) == self.num_of_average:
            self.valence_all.pop(0)
            self.arousal_all.pop(0)
            self.fun_status.pop(0)
            self.difficulty_status.pop(0)
            self.immersion_status.pop(0)
            self.emotion_status.pop(0)

        # Analyze result sum
        self.arousal_all.append(class_ar)
        self.valence_all.append(class_va)
        self.fun_status.append(fun)
        self.difficulty_status.append(difficulty)
        self.immersion_status.append(immersion)
        self.emotion_status.append(emotion)



        # draw graph
        d = {
            'eeg_realtime': eeg_realtime[:, self.max_seq_length - 1],
            'arousal_all': np.array(self.arousal_all),
            'valence_all': np.array(self.valence_all),
            # 'fun_stat': self.fun_stat_dict[self.final_fun],
            # 'immersion_stat': self.immersion_stat_dict[self.final_immersion],
            # 'difficulty_stat': self.difficulty_stat_dict[self.final_difficulty],
            # 'emotion_stat': self.emotion_stat_dict[self.final_emotion],
            'fun_stat': self.fun_stat_dict[fun_stat_code],
            'immersion_stat': self.immersion_stat_dict[immersion_stat_code],
            'difficulty_stat': self.difficulty_stat_dict[difficulty_stat_code],
            'emotion_stat': self.emotion_stat_dict[emotion_stat_code],
            'fun_stat_record': self.counter(self.fun_records, self.fun_stat_dict),
            'immersion_stat_record': self.counter(self.immersion_records, self.immersion_stat_dict),
            'difficulty_stat_record': self.counter(self.difficulty_records, self.difficulty_stat_dict),
            'emotion_stat_record': self.counter(self.emotion_records, self.emotion_stat_dict),
            'record_duration': self.record_duration,
            'fun_status': self.fun_status,
            'immersion_status': self.immersion_status,
            'difficulty_status': self.difficulty_status,
            'emotion_status': self.emotion_status,
            'final_score_pred': self.final_score_pred
        }

        return d

    def most_common(self, target_list: list, last_status: int):
        """ Find most common

        :param target_list:
        :param last_status:
        :return:
        """
        a = Counter(target_list).most_common(2)
        final_status = int(a[0][0])
        if len(a) != 1 and a[0][1] == a[1][1]:
            if int(a[0][0]) == last_status or int(a[1][0]) == last_status:
                final_status = last_status
        return final_status

    def counter(self, data: list, data_dict: dict):
        """ Counts the duplicate elements

        :param data:
        :param data_dict:
        :return:
        """
        counter_dict = dict()
        data = [str(d) for d in data]
        data = Counter(data)
        for k, v in data_dict.items():
            counter_dict[v] = data[str(k)]

        return counter_dict

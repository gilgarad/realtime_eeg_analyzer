import numpy as np
from realtime_eeg.emotiv import Emotiv
import threading
from socketIO_client import SocketIO, LoggingNamespace
from models.fft_convention import FFTConvention
import json
import re
from os import listdir
from os.path import join, isfile
from datetime import datetime
# from utils.live_plot import draw_graph, Communicate, CustomMainWindow
from utils.live_plot import Communicate, CustomMainWindow
from utils.similarity import Similarity
from PyQt5 import QtWidgets
import sys
import random
from collections import Counter
from keras.models import model_from_json
import keras.backend.tensorflow_backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RealtimeEmotion:
    def __init__(self, path="./Training Data/", realtime=False, save_path=None):
        if realtime:
            self.emotiv = Emotiv()
        self.fft_conv = FFTConvention(path=path)
        self.socket_port = 8080
        self.save_path = save_path

        self.path = path

    def load_model(self, path):
        # Load Saved Model
        with open(join(path, 'model_json_multiloss4_resnet18_fftstd_3class.json'), 'r') as f:
            loaded_model_json = f.read()
        with K.tf.device('/cpu:0'):
            loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(join(path, 'model_weights_multiloss4_resnet18_fftstd_3class.h5'))

        losses = {'amusement': 'categorical_crossentropy',
                  'immersion': 'categorical_crossentropy',
                  'difficulty': 'categorical_crossentropy',
                  'emotion': 'categorical_crossentropy'}
        loss_weights = {'amusement': 1.0, 'immersion': 1.0, 'difficulty': 1.0, 'emotion': 1.0}
        loaded_model.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['accuracy'])
        self.model = loaded_model
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
        y_pred = self.model.predict(x=x_test, batch_size=1)

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

    def send_result_to_application(self, emotion_class):
        """
        Send emotion predict to web app.
        Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        Output: Send emotion prediction to web app.
        """
        socket = SocketIO('localhost', self.socket_port, LoggingNamespace)
        socket.emit('realtime emotion', emotion_class)

    def run_process(self, addData_callbackFunc, get_record_status, get_subject_name):

        mySrc = Communicate()
        mySrc.data_signal.connect(addData_callbackFunc)

        self.load_model(self.path)
        self.emotiv.subscribe()

        number_of_channel = 14
        sampling_rate = 128
        count = 0
        realtime_eeg_in_second = 5  # Realtime each ... seconds
        number_of_realtime_eeg = sampling_rate * realtime_eeg_in_second

        channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        channel_names = np.array(channel_names)

        # original
        eeg_realtime = np.zeros((number_of_channel, number_of_realtime_eeg), dtype=np.double)

        response_records = list()
        fun_records = list()
        immersion_records = list()
        difficulty_records = list()
        emotion_records = list()

        time_counter = 0
        record_start_time = 0
        final_emotion = 5
        final_emotion2 = 0
        final_fun = 0
        final_immersion = 0
        final_difficulty = 0

        record_status = False
        connection_status = 0
        disconnected_list = list()
        num_frame_check = 16
        num_of_average = int(sampling_rate / num_frame_check)
        arousal_all = [2.0] * num_of_average
        valence_all = [2.0] * num_of_average
        fun_status = [0] * num_of_average * realtime_eeg_in_second
        immersion_status = [0] * num_of_average * realtime_eeg_in_second
        difficulty_status = [0] * num_of_average * realtime_eeg_in_second
        emotion_status = [0] * num_of_average * realtime_eeg_in_second

        fun_stat_dict = {
            0: 'Boring',
            1: 'Neutral',
            2: 'Fun'
        }

        immersion_stat_dict = {
            0: 'Distracted',
            1: 'Immersed'
        }

        difficulty_stat_dict = {
            0: 'Easy',
            1: 'Difficult'
        }

        emotion_stat_dict = {
            0: 'Happy',
            1: 'Neutral',
            2: 'Annoyed'
        }

        # Try to get if it has next step
        while self.emotiv.is_run:
            res = self.emotiv.retrieve_packet()
            current_time = datetime.now()
            # print(res)
            #     res_result.append(res)

            if 'eeg' in res:
                new_data = res['eeg'][3: 3 + number_of_channel]
                eeg_realtime = np.insert(eeg_realtime, number_of_realtime_eeg, new_data, axis=1)
                eeg_realtime = np.delete(eeg_realtime, 0, axis=1)

                if record_status:
                    new_data = [current_time]
                    new_data.append(res['time'])
                    new_data.extend(res['eeg'])
                    # print(new_data)
                    response_records.append(new_data)

                count += 1

                # print('eeg')
            elif 'dev' in res:
                # signal quality 0 None, 1 bad to 4 good
                cnt = 0
                disconnected_list = list()
                for idx, i in enumerate(res['dev'][2]):
                    if i > 0:
                        cnt += 1
                    else:
                        disconnected_list.append(idx)
                connection_status = int(float(cnt / 14) * 100)

                # print('connection status:', connection_status)
                # print(res)

            elif 'error' in res:
                print(res)

                break
            else:
                # print(res)
                continue
                # break

            if count % num_frame_check == 0:
                emotion_class, class_ar, class_va, fun, difficulty, immersion, emotion = self.analyze_eeg_data(eeg_realtime)

                fun_status.pop(0)
                difficulty_status.pop(0)
                immersion_status.pop(0)
                emotion_status.pop(0)

                fun_status.append(fun)
                difficulty_status.append(difficulty)
                immersion_status.append(immersion)
                emotion_status.append(emotion)

                if len(valence_all) == num_of_average:
                    valence_all.pop(0)
                    arousal_all.pop(0)

                # temp
                arousal_all.append(class_ar)
                valence_all.append(class_va)
                # draw graph
                d = eeg_realtime[:, number_of_realtime_eeg - 1]
                # d = np.concatenate([[final_emotion], [connection_status]], axis=0)
                # d = np.concatenate([[final_emotion], [connection_status], [disconnected_list]], axis=0)
                d = {
                    'eeg_realtime': d,
                    'final_emotion': final_emotion,
                    'connection_status': connection_status,
                    'disconnected_list': channel_names[disconnected_list],
                    'arousal_all': np.array(arousal_all),
                    'valence_all': np.array(valence_all),
                    'fun_stat': fun_stat_dict[final_fun],
                    'immersion_stat': immersion_stat_dict[final_immersion],
                    'difficulty_stat': difficulty_stat_dict[final_difficulty],
                    'emotion_stat': emotion_stat_dict[final_emotion2],
                    'fun_stat_record': self.counter(fun_records, fun_stat_dict),
                    'immersion_stat_record': self.counter(immersion_records, immersion_stat_dict),
                    'difficulty_stat_record': self.counter(difficulty_records, difficulty_stat_dict),
                    'emotion_stat_record': self.counter(emotion_records, emotion_stat_dict),
                    'fun_records': fun_records,
                    'immersion_records': immersion_records,
                    'difficulty_records': difficulty_records,
                    'emotion_records': emotion_records,
                    'record_start_time': record_start_time
                }
                mySrc.data_signal.emit(d)

            # print('Record status %r' % get_record_status())
            if get_record_status() and record_status is False:
                record_status = True
                response_records = list()
                emotion_records = list()
                fun_records = list()
                difficulty_records = list()
                immersion_records = list()
                record_start_time = datetime.now()
            elif not get_record_status() and record_status is True:
                record_status = False
                response_records = self.save_data(data=response_records, save_path=self.save_path,
                                         filename=get_subject_name(), time_counter=time_counter)
                record_start_time = 0
                time_counter += 1

            if count == sampling_rate:
                emotion_dict = {
                    1: "fear - nervous - stress - tense - upset",
                    2: "happy - alert - excited - elated",
                    3: "relax - calm - serene - contented",
                    4: "sad - depressed - lethargic - fatigue",
                    5: "neutral"
                }
                class_ar = np.round(np.mean(arousal_all))
                class_va = np.round(np.mean(valence_all))

                emotion_class = self.fft_conv.determine_emotion_class(class_ar, class_va)
                final_emotion = emotion_class
                final_emotion2 = self.most_common(emotion_status, final_emotion2)
                final_difficulty = self.most_common(difficulty_status, final_difficulty)
                final_fun = self.most_common(fun_status, final_fun)
                final_immersion = self.most_common(immersion_status, final_immersion)

                if record_status:
                    emotion_records.append(final_emotion2)
                    fun_records.append(final_fun)
                    difficulty_records.append(final_difficulty)
                    immersion_records.append(final_immersion)

                count = 0

        self.emotiv.ws.close()

    def most_common(self, target_list, last_status):
        a = Counter(target_list).most_common(2)
        final_status = int(a[0][0])
        if len(a) != 1 and a[0][1] == a[1][1]:
            if int(a[0][0]) == last_status or int(a[1][0]) == last_status:
                final_status = last_status
        return final_status

    def counter(self, data, data_dict):
        counter_dict = dict()
        data = [str(d) for d in data]
        data = Counter(data)
        for k, v in data_dict.items():
            counter_dict[v] = data[str(k)]

        return counter_dict

    def extract_channels(self, signals):
    
        signals = signals[:, 7:]
    
        return signals

    def run_process2(self, test_path):
        #
        number_of_channel = 14
        sampling_rate = 128
        time_interval = 0.001
        show_interval = 10
        is_first = True
        num_channels = 14
        realtime_eeg_in_second = 5  # Realtime each ... seconds
        number_of_realtime_eeg = sampling_rate * realtime_eeg_in_second
    
        eeg_from_file = np.load(test_path) # shape: number_of_realtime_eeg * 20(EEG channels + alpha)
    
        if eeg_from_file.shape[1] != 14:
            eeg = self.extract_channels(eeg_from_file)
        else:
            eeg = eeg_from_file
    
        eeg = eeg.T
    
        for i in range(0, eeg.shape[1], number_of_realtime_eeg):
            self.analyze_eeg_data(eeg[:,i:i+number_of_realtime_eeg])

    def save_data(self, data, save_path, filename, time_counter):
        # print(save_path)
        if save_path is not None:
            # np.save(join(save_path, str(time_counter)), data)
            np.save(join(save_path, filename), data)

        return list()


def draw_graph(run_process=None):
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    # myGUI = CustomMainWindow(run_process)
    myGUI = CustomMainWindow()
    myDataLoop = threading.Thread(name='myDataLoop', target=run_process, daemon=True,
                                  args=(myGUI.addData_callbackFunc, myGUI.get_record_status, myGUI.get_subject_name))
    myDataLoop.start()

    sys.exit(app.exec_())


if __name__ == '__main__':
    # print('First Argument', sys.argv[1])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--realtime', type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument('--test_path', type=str, default=None)
    # args = parser.parse_args()
    # realtime = args.realtime
    # test_path = args.test_path
    config = json.load(open(join('.', 'config', 'system_config.json')))
    realtime = config['realtime']
    test_path = config['test_path']
    save_path = config['save_path']
    # print(realtime)
    # print(test_path)


    # print("Starting webapp...")
    # threading.Thread(target=execute_js, args=('./webapp/index.js', )).start()
    # success = execute_js('./webapp/index.js')

    print("Starting realtime emotion engine...")
    realtime_emotion = RealtimeEmotion(realtime=realtime, save_path=save_path)
    if realtime:
        # realtime_emotion.run_process()
        draw_graph(run_process=realtime_emotion.run_process)
    else:
        realtime_emotion.run_process2(test_path=test_path)





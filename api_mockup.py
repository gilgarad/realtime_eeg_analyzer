from const import *
from webapp.webapp import app, socketio, send_to_html, set_status_controller
import threading
import numpy as np
from dao_models.subject import Subject
from dao_models.status_controller import StatusController
from dao_models.trial import Trial

import time
from collections import Counter
from datetime import datetime


app.config['SECRET_KEY'] = 'secret!'


# chart test
class SendMockup:
    def __init__(self):
        self.status_controller = StatusController()
        self.trial = Trial()
        self.subject = Subject()

        self.num_frame_check = 16
        self.sampling_rate = 128
        self.realtime_eeg_in_second = 3  # Realtime each ... seconds
        self.eeg_seq_length = self.sampling_rate * self.realtime_eeg_in_second
        self.max_seq_length = self.sampling_rate * 60 * 5  #
        self.num_of_average = int(self.sampling_rate / self.num_frame_check)

        self.arousal_all = [2.0] * self.num_of_average
        self.valence_all = [2.0] * self.num_of_average
        self.fun_status = [0] * self.num_of_average
        self.immersion_status = [0] * self.num_of_average
        self.difficulty_status = [0] * self.num_of_average
        self.emotion_status = [0] * self.num_of_average

        self.number_of_channel = 14

        self.eeg_realtime = np.zeros(shape=(self.number_of_channel, self.max_seq_length))

        self.emotion_records = list()
        self.fun_records = list()
        self.difficulty_records = list()
        self.immersion_records = list()
        self.record_start_time = 0
        self.record_duration = 0
        self.final_score_pred = np.zeros((4, 1))

    def send_mock_data(self, transmit_data):
        fun_stat_dict = {0: '일반', 1: '재미있음'}
        immersion_stat_dict = {0: '일반', 1: '몰입됨'}
        difficulty_stat_dict = {0: '쉬움', 1: '어려움'}
        emotion_stat_dict = {0: '즐거움', 1: '일반', 2: '짜증'}

        while True:
            data = dict()

            if self.status_controller.headset_status == 1:
                self.status_controller.headset_status = 2
                self.status_controller.set_electrodes_connection([4] * 14)
            elif self.status_controller.headset_status == 0:
                self.status_controller.set_electrodes_connection([0] * 14)
                transmit_data(send_type=1)
                time.sleep(0.1)
                continue

            if self.status_controller.headset_status == 2:
                is_connected = True
            else:
                is_connected = False

            data['is_connected'] = is_connected
            data['analyze_status'] = self.status_controller.analyze_status
            data['connection_status'] = self.status_controller.connection_status

            # Set some random variable for mockup
            # fun_stat = 0
            # immersion_stat = 0
            # difficulty_stat = 0
            # emotion_stat = 0
            # class_ar = 0
            # class_va = 0
            fun_stat = int(round((np.sin(np.random.randint(100)) + 1) * 4 + 1))
            immersion_stat = int(round((np.sin(np.random.randint(100)) + 1) * 4 + 1))
            difficulty_stat = int(round((np.sin(np.random.randint(100)) + 1) * 4 + 1))
            emotion_stat = int(round((np.sin(np.random.randint(100)) + 1) * 4 + 1))
            class_ar = int(round((np.sin(np.random.randint(100)) + 1) * 1 + 1))
            class_va = int(round((np.sin(np.random.randint(100)) + 1) * 1 + 1))

            self.fun_status.append(fun_stat)
            self.immersion_status.append(immersion_stat)
            self.difficulty_status.append(difficulty_stat)
            self.emotion_status.append(emotion_stat)
            self.arousal_all.append(class_ar)
            self.valence_all.append(class_va)
            self.fun_status.pop(0)
            self.immersion_status.pop(0)
            self.difficulty_status.pop(0)
            self.emotion_status.pop(0)
            self.valence_all.pop(0)
            self.arousal_all.pop(0)

            # stat_code
            if fun_stat >= 6:
                fun_stat_code = 1
            else:
                fun_stat_code = 0

            if immersion_stat >= 5:
                immersion_stat_code = 1
            else:
                immersion_stat_code = 0

            if difficulty_stat >= 5:
                difficulty_stat_code = 1
            else:
                difficulty_stat_code = 0

            if emotion_stat > 6:
                emotion_stat_code = 0
            elif emotion_stat >= 4:
                emotion_stat_code = 1
            else:
                emotion_stat_code = 2

            if self.status_controller.analyze_status == 3:
                self.emotion_records = list()
                self.fun_records = list()
                self.difficulty_records = list()
                self.immersion_records = list()
                self.record_start_time = 0
                self.record_duration = 0
                self.final_score_pred = np.zeros((4, 1))
                self.status_controller.analyze_status = 0

            elif self.status_controller.analyze_status == 1:
                self.fun_records.append(fun_stat_code)
                self.immersion_records.append(immersion_stat_code)
                self.difficulty_records.append(difficulty_stat_code)
                self.emotion_records.append(emotion_stat_code)
                if self.record_start_time == 0:
                    self.record_start_time = datetime.now()
                self.record_duration = (datetime.now() - self.record_start_time).seconds
            elif self.status_controller.analyze_status == 2:
                self.final_score_pred = np.ones(shape=(4, 1, 1))
                # print(self.final_score_pred)

            data['fun_stat'] = fun_stat_dict[fun_stat_code]
            data['immersion_stat'] = immersion_stat_dict[immersion_stat_code]
            data['difficulty_stat'] = difficulty_stat_dict[difficulty_stat_code]
            data['emotion_stat'] = emotion_stat_dict[emotion_stat_code]
            data['fun_status'] = self.fun_status
            data['immersion_status'] = self.immersion_status
            data['difficulty_status'] = self.difficulty_status
            data['emotion_status'] = self.emotion_status

            data['final_score_pred'] = self.final_score_pred
            data['arousal_all'] = self.arousal_all
            data['valence_all'] = self.valence_all
            data['record_duration'] = self.record_duration
            data['fun_stat_record'] = self.counter(self.fun_records, fun_stat_dict)
            data['immersion_stat_record'] = self.counter(self.immersion_records, immersion_stat_dict)
            data['difficulty_stat_record'] = self.counter(self.difficulty_records, difficulty_stat_dict)
            data['emotion_stat_record'] = self.counter(self.emotion_records, emotion_stat_dict)
            data['final_score_pred'] = self.final_score_pred

            data['eeg_realtime'] = self.eeg_realtime[:, self.max_seq_length - 1]

            t = threading.Thread(target=transmit_data, args=(0, data,))
            t.start()

            t = threading.Thread(target=transmit_data, args=(1,))
            t.start()

            time.sleep(1)

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


if __name__ == '__main__':
    send_chart = SendMockup()
    set_status_controller(send_chart.status_controller, send_chart.subject,
                          send_chart.trial)

    th = threading.Thread(name='myDataLoop', target=send_chart.send_mock_data, daemon=True,
                          args=(send_to_html,))
    th.start()
    socketio.run(app, host=HOST, port=PORT, debug=True)
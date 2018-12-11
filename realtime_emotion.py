import numpy as np
from realtime_eeg.emotiv import Emotiv
import threading
from socketIO_client import SocketIO, LoggingNamespace
from Naked.toolshed.shell import execute_js
import argparse
from models.fft_convention import FFTConvention
import json
import re
from os import listdir
from os.path import join, isfile
from datetime import datetime
# from utils.live_plot import draw_graph, Communicate, CustomMainWindow
from utils.live_plot import Communicate, CustomMainWindow
from PyQt5 import QtWidgets
import sys


class RealtimeEmotion:
    def __init__(self, path="./Training Data/", realtime=False, save_path=None):
        if realtime:
            self.emotiv = Emotiv()
        self.fft_conv = FFTConvention(path=path)
        self.socket_port = 8080
        self.save_path = save_path

    def process_all_data(self, all_channel_data):
        """
        Process all data from EEG data to predict emotion class.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model. And send it to web ap
        """
        emotion_class = self.fft_conv.get_emotion(all_channel_data)

        return emotion_class

    def send_result_to_application(self, emotion_class):
        """
        Send emotion predict to web app.
        Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        Output: Send emotion prediction to web app.
        """
        socket = SocketIO('localhost', self.socket_port, LoggingNamespace)
        socket.emit('realtime emotion', emotion_class)

    def run_process(self, addData_callbackFunc):

        mySrc = Communicate()
        mySrc.data_signal.connect(addData_callbackFunc)

        self.emotiv.subscribe()

        # 9. Retrieve EEG
        number_of_channel = 14
        sampling_rate = 128
        count = 0
        realtime_eeg_in_second = 5  # Realtime each ... seconds
        number_of_realtime_eeg = sampling_rate * realtime_eeg_in_second

        channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        # Graph init
        # plt.ion()
        # fig = plt.figure(figsize=(13, 6))
        # ax = fig.add_subplot(111)

        # original
        threads = []
        eeg_realtime = np.zeros((number_of_channel, number_of_realtime_eeg), dtype=np.double)

        res_all = list()

        time_counter = 0
        last_time = datetime.now()
        current_step = -1
        final_emotion = 5

        # Try to get if it has next step
        while self.emotiv.is_run:
            res = self.emotiv.retrieve_packet()
            current_time = datetime.now()
            # print(res)
            #     res_result.append(res)

            if 'eeg' in res:
                # res_result.append(res)
                if current_step + 1 == int(res['eeg'][0]):
                    current_step = int(res['eeg'][0])
                else:
                    # ignore this part
                    # print(current_step, int(res['eeg'][0]))
                    continue

                if current_step == 127:
                    current_step = -1

                new_data = res['eeg'][3: 3 + number_of_channel]
                eeg_realtime = np.insert(eeg_realtime, number_of_realtime_eeg, new_data, axis=1)
                eeg_realtime = np.delete(eeg_realtime, 0, axis=1)

                new_data = [current_time]
                new_data.append(res['time'])
                new_data.extend(res['eeg'])
                # print(new_data)
                res_all.append(new_data)

                count += 1

                # print('eeg')
            elif 'dev' in res:
                # signal quality 0 None, 1 bad to 4 good
                cnt = 0
                for i in res['dev'][2]:
                    if i > 0:
                        cnt += 1
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

            if count % 16 == 0:
                # draw graph
                d = eeg_realtime[:, number_of_realtime_eeg - 1]
                d = np.concatenate([d, [final_emotion]], axis=0)
                mySrc.data_signal.emit(d)

            if count == sampling_rate:
                emotion_class = self.process_all_data(eeg_realtime)
                emotion_dict = {
                    1: "fear - nervous - stress - tense - upset",
                    2: "happy - alert - excited - elated",
                    3: "relax - calm - serene - contented",
                    4: "sad - depressed - lethargic - fatigue",
                    5: "neutral"
                }
                # final_emotion = emotion_dict[emotion_class]
                final_emotion = emotion_class
                print(emotion_dict[emotion_class])

                # send emotion_class to web app
                t = threading.Thread(target=self.send_result_to_application, args=(emotion_class,))
                threads.append(t)
                t.start()
                count = 0
                # count -= 1

            if (current_time - last_time).seconds == 60:
                time_counter += 1
                # print(current_time, last_time)
                last_time = current_time
                res_all = self.save_data(data=res_all, save_path=self.save_path, time_counter=time_counter)

            # if len(res_result) == sampling_rate:
            #     y1 = [r['eeg'][3] for r in res_result]
            #     x1 = [r['time'] % show_interval for r in res_result]
            #     res_result.pop(0)
            #
            #     if is_first:
            #         line1, = ax.plot(x1, y1, 'b-')
            #         # plt.show()
            #         is_first = False
            #     else:
            #         line1.set_ydata(y1)
            #
            #     plt.pause(0.001)
            #
            #     # fig.canvas.draw()
            #     # plt.plot(x1, y1)
            #     # plt.show()

        self.emotiv.ws.close()

    def extract_channels(signals):
    
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
            self.process_all_data(eeg[:,i:i+number_of_realtime_eeg])

    def save_data(self, data, save_path, time_counter):
        # print(save_path)
        if save_path is not None:
            np.save(join(save_path, str(time_counter)), data)

        return list()


def draw_graph(run_process=None):
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    # myGUI = CustomMainWindow(run_process)
    myGUI = CustomMainWindow()
    myDataLoop = threading.Thread(name='myDataLoop', target=run_process, daemon=True, args=(myGUI.addData_callbackFunc,))
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





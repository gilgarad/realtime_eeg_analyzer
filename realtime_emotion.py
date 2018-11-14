import numpy as np
from realtime_eeg.emotiv import Emotiv
import threading
from socketIO_client import SocketIO, LoggingNamespace
import argparse
from models.fft_convention import FFTConvention


class RealtimeEmotion:
    def __init__(self, path="./Training Data/", realtime=False):
        if realtime:
            self.emotiv = Emotiv()
        self.fft_conv = FFTConvention(path=path)
        self.socket_port = 8080

    def process_all_data(self, all_channel_data):
        """
        Process all data from EEG data to predict emotion class.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model. And send it to web ap
        """
        emotion_class = self.fft_conv.get_emotion(all_channel_data)
        emotion_dict = {
            1: "fear - nervous - stress - tense - upset",
            2: "happy - alert - excited - elated",
            3: "relax - calm - serene - contented",
            4: "sad - depressed - lethargic - fatigue",
            5: "neutral"
        }
        print(emotion_dict[emotion_class])

        # send emotion_class to web app
        self.send_result_to_application(emotion_class)


    def send_result_to_application(self, emotion_class):
        """
        Send emotion predict to web app.
        Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        Output: Send emotion prediction to web app.
        """
        socket = SocketIO('localhost', self.socket_port, LoggingNamespace)
        socket.emit('realtime emotion', emotion_class)

    def run_process(self):

        self.emotiv.subscribe()

        # 9. Retrieve EEG
        number_of_channel = 14
        sampling_rate = 128
        time_interval = 0.001
        show_interval = 10
        is_first = True
        num_channels = 14
        count = 0
        realtime_eeg_in_second = 5  # Realtime each ... seconds
        number_of_realtime_eeg = sampling_rate * realtime_eeg_in_second

        channel_names = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4"
        ]



        # Graph init
        # plt.ion()
        # fig = plt.figure(figsize=(13, 6))
        # ax = fig.add_subplot(111)

        # original
        threads = []
        eeg_realtime = np.zeros((number_of_channel, number_of_realtime_eeg), dtype=np.double)
        init = True

        # initial
        res_result = list()
        last_res = None
        while last_res is None:
            last_res = self.emotiv.retrieve_packet()
            if 'eeg' not in last_res:
                last_res = None
            else:
                new_data = last_res['eeg'][3: 3 + number_of_channel]
                for idx, data in enumerate(new_data):
                    eeg_realtime[idx, 0] = data

        # Try to get if it has next step
        while self.emotiv.is_run:
            res = self.emotiv.retrieve_packet()
            #     print(res)
            #     res_result.append(res)
            if 'eeg' in res and res['eeg'][0] != last_res['eeg'][0]:
                # res_result.append(res)
                last_res = res

                new_data = res['eeg'][3: 3 + number_of_channel]
                eeg_realtime = np.insert(eeg_realtime, number_of_realtime_eeg, new_data, axis=1)
                eeg_realtime = np.delete(eeg_realtime, 0, axis=1)

                count += 1

                # print(prev_step)
            else:
                print(res)
                continue
                # break
            # time.sleep(time_interval)

            if count == sampling_rate:

                t = threading.Thread(target=self.process_all_data, args=(eeg_realtime,))
                threads.append(t)
                t.start()
                count = 0

                # break


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


    def run_process2(self, test_path):
        #
        pass


if __name__ == '__main__':
    # print('First Argument', sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--test_path', type=str, default=None)
    args = parser.parse_args()
    realtime = args.realtime
    test_path = args.test_path


    print("Starting webapp...")
    # success = execute_js('./webapp/index.js')

    print("Starting realtime emotion engine...")
    realtime_emotion = RealtimeEmotion(realtime=realtime)
    if realtime:
        realtime_emotion.run_process()
    else:
        realtime_emotion.run_process2(test_path=test_path)

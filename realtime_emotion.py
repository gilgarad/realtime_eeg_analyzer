from socketIO_client import SocketIO, LoggingNamespace
from os.path import join
import time
import threading
import numpy as np

# Import implemented parts
from const import *
from system_shares import logger
from webapp.webapp import app, socketio, send_to_html, set_status_controller

from realtime_eeg.headset.emotiv import Emotiv
from realtime_eeg.analyze_eeg import AnalyzeEEG
from dao_models.subject import Subject
from dao_models.status_controller import StatusController
from dao_models.trial import Trial

# Tensorflow debug Log off
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RealtimeEEGAnalyzer:
    def __init__(self, path="./Training Data/", save_path=None):

        self.path = path
        self.socket_port = 8080
        self.save_path = save_path

        self.emotiv = Emotiv()
        self.status_controller = StatusController()
        self.subject = Subject()
        self.trial = Trial()
        self.analyze_eeg = AnalyzeEEG(self.path)
        self.models = list()

    def connect_headset(self):
        self.emotiv.connect_headset()
        self.emotiv.login()
        self.emotiv.authorize()
        self.emotiv.query_headsets()
        self.emotiv.close_old_sessions()
        self.emotiv.create_session()
        self.emotiv.subscribe()

    def disconnect_headset(self):
        self.emotiv.unsubscribe()
        # self.emotiv.close_old_sessions()
        # self.emotiv.logout()

    def load_model(self, path):

        # model1 = 'model_json_multiloss4_resnet18_fftstd_3class.json'
        # weight1 = 'model_weights_multiloss4_resnet18_fftstd_3class.h5'
        model1 = 'model_json_multiloss4_3seconds_resnet18_fftstd_2class_type1.json'
        weight1 = 'model_weights_multiloss4_3seconds_resnet18_fftstd_2class_type1.h5'
        model2 = 'model_json_multiloss4_3seconds_resnet18_fftstd_2class_type2.json'
        weight2 = 'model_weights_multiloss4_3seconds_resnet18_fftstd_2class_type2.h5'
        model3 = 'model_json_multiloss4_attention_score.json'
        weight3 = 'model_weights_multiloss4_attention_score.h5'

        model_list = [[model1, weight1], [model2, weight2], [model3, weight3]]

        self.analyze_eeg.load_models(model_names=model_list)

    def send_result_to_application(self, emotion_class):
        """
        Send emotion predict to web app.
        Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        Output: Send emotion prediction to web app.
        """
        socket = SocketIO('localhost', self.socket_port, LoggingNamespace)
        socket.emit('realtime emotion', emotion_class)

    def run_process(self, transmit_data=None):

        # Load models to use for prediction
        self.load_model(self.path)

        threads = list()

        # Try to get if it has next step
        while self.emotiv.is_run:
            # 1. Headset connection / disconnection
            if not self.emotiv.is_connect and self.status_controller.headset_status == 0:
                # wait until connected
                transmit_data(send_type=1)
                time.sleep(0.1)
                continue
            elif not self.emotiv.is_connect and self.status_controller.headset_status == 1:
                # connect
                # TODO thread for transmit_data until headset_status changed
                transmit_data(send_type=1)
                self.connect_headset()
                if self.status_controller.headset_status == 1:
                    self.status_controller.headset_status += 1

            elif self.emotiv.is_connect and self.status_controller.headset_status == 0:
                # disconnect
                print('disconnect !!')
                self.disconnect_headset()

            # 2. Command status control
            analyze_status = self.status_controller.analyze_status
            if analyze_status == 3:
                self.save_data(data=self.analyze_eeg.response_records, save_path=self.save_path,
                               filename=self.trial.trial_name)
                self.status_controller.analyze_status = 0
            self.analyze_eeg.set_record_status(analyze_status)

            # 3. Retrieve Info from headset
            res = self.emotiv.retrieve_packet()

            # 4. Process retrieved info from headset
            if 'eeg' in res:
                self.analyze_eeg.store_eeg_rawdata(rawdata=res)

                # print('eeg')
            elif 'dev' in res:
                # signal quality 0 None, 1 bad to 4 good
                self.status_controller.set_electrodes_connection(res['dev'][2])
                # print('connection status:', connection_status)
                # print(res)
            elif 'error' in res:
                print(res)
                break
            else:
                # print(res)
                continue
                # break

            # 5. Send info to ui
            if self.analyze_eeg.count % self.analyze_eeg.num_frame_check == 0:
                # print('Send frame:', self.analyze_eeg.count)
                d = self.analyze_eeg.analyze_and_evaluate_moment()
                d['connection_status'] = self.status_controller.connection_status
                d['disconnected_list'] = self.status_controller.disconnected_list
                d['is_connected'] = self.emotiv.is_connect

                # send data
                # transmit_data(send_type=0, data=d)
                t = threading.Thread(target=transmit_data, args=(0, d,))
                t.start()
                threads.append(t)

                # send headset connection status
                # transmit_data(send_type=1)
                t = threading.Thread(target=transmit_data, args=(1,))
                t.start()
                threads.append(t)

            # print('Record status %r' % self.status_controller.analyze_status)

        self.emotiv.ws.close()

    def extract_channels(self, signals):
        signals = signals[:, 7:]
        return signals

    def run_process2(self, test_path):
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
            self.analyze_eeg.analyze_eeg_data(eeg[:,i:i+number_of_realtime_eeg])

    def save_data(self, data, save_path, filename):
        # print(save_path)
        if save_path is not None and filename is not None and filename != '':
            np.save(join(save_path, filename), data)


if __name__ == '__main__':
    # print('First Argument', sys.argv[1])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--realtime', type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument('--test_path', type=str, default=None)
    # args = parser.parse_args()
    # realtime = args.realtime
    # test_path = args.test_path
    config = json.load(open(join('.', 'config', 'system_config.json')))
    test_path = config['test_path']
    save_path = config['save_path']
    # print(test_path)

    # print("Starting webapp...")
    # threading.Thread(target=execute_js, args=('./webapp/index.js', )).start()
    # success = execute_js('./webapp/index.js')

    logger.info("Starting realtime emotion engine...")

    realtime_eeg_analyzer = RealtimeEEGAnalyzer(save_path=save_path)
    set_status_controller(realtime_eeg_analyzer.status_controller, realtime_eeg_analyzer.subject,
                          realtime_eeg_analyzer.trial)

    th = threading.Thread(name='myDataLoop', target=realtime_eeg_analyzer.run_process, daemon=True,
                          args=(send_to_html, ))
    th.start()
    # draw_graph(run_process=realtime_eeg_analyzer.run_process)

    # app.run(host=HOST, port=PORT, debug=True, threaded=False)
    socketio.run(app, host=HOST, port=PORT, debug=True)

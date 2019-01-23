from flask import Flask, request, render_template
from flask_socketio import SocketIO
from os.path import join
import numpy as np
from datetime import datetime

# app = Flask(__name__, template_folder='mdb_free_4.6.1')
app = Flask(__name__)
socketio = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'
# Bootstrap(app)
# app.config['BOOTSTRAP_SERVE_LOCAL'] = True #This turns file serving static
# api = Api(app)
realtime_emotion2 = list()

is_on_status = [False, False, None]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    # return render_template('index2.html', values=values, labels=labels)
    # return render_template('../mdb_free_4.6.1/index.html')


@socketio.on('update_data')
def test_message(data):
    # print('transmit!!')
    # data = {'username': 'isaacsim',
    #      'data': 'nothing'}
    text_display_analysis = '일반분석 (5초 평균 우세)' \
                            + '\n재미: ' + data['fun_stat'] \
                            + '\n몰입감: ' + data['immersion_stat'] \
                            + '\n난이도: ' + data['difficulty_stat'] \
                            + '\n감정: ' + data['emotion_stat']

    if data['is_analysis']:
        total_played_time = (datetime.now() - data['record_start_time']).seconds
        text_display_analysis = text_display_analysis \
                                + '\n------------------------------------------------------------\n' \
                                + '게임 분석 중\n' \
                                + '플레이시간: ' + str(total_played_time) + 's' \
                                + '\n재미: ' + make_analysis_text(data['fun_stat_record'], total_played_time) \
                                + '\n몰입감: ' + make_analysis_text(data['immersion_stat_record'], total_played_time) \
                                + '\n난이도: ' + make_analysis_text(data['difficulty_stat_record'], total_played_time) \
                                + '\n감정: ' + make_analysis_text(data['emotion_stat_record'], total_played_time)

    emotion_mean = 1 - np.mean(data['emotion_status'], axis=0)
    if emotion_mean < 0:
        emotion_mean = 0

    data = {
        'eeg_mean': np.mean(data['eeg_realtime'], axis=0),
        'eeg_channels': data['eeg_realtime'].tolist(),
        'is_connected': data['is_connected'],
        'connection_status': data['connection_status'],
        'arousal_mean': np.mean(data['arousal_all'], axis=0),
        'valence_mean': np.mean(data['valence_all'], axis=0),
        'fun_mean': np.mean(data['fun_status'], axis=0),
        'immersion_mean': np.mean(data['immersion_status'], axis=0),
        'difficulty_mean': np.mean(data['difficulty_status'], axis=0),
        'emotion_mean': emotion_mean,
        'analysis': text_display_analysis
    }

    socketio.emit('response', data, json=True, namespace='/update_data')


@socketio.on('connect_headset')
def connect_headset(message):
    # print('request in!!')
    # print(message)
    # realtime_emotion2[0].connect_headset()
    # print(is_connection_request)
    is_on_status[0] = not get_connection_request()


def set_realtime_emotion(rm):
    realtime_emotion2.append(rm)


def get_connection_request():
    return is_on_status[0]


@socketio.on('control_analysis')
def start_analysis(message):
    # print(message)

    if message['stat'] != 2:
        is_on_status[1] = not get_analysis_status()
        is_on_status[2] = message['data']



def get_analysis_status():
    return is_on_status[1]


def get_subject_name():
    return is_on_status[2]


def make_analysis_text(data, duration=0):
    """
    Make analysis details with percentage information and its duration for each component

    :param data:
    :return:
    """
    txt = ''

    total_count = 0
    for k, v in data.items():
        total_count += v

    for k, v in data.items():
        if total_count == 0:
            percentage = '0'
            final_duration = '0'
        else:
            percentage = str(int(v * 100 / total_count))
            final_duration = str(int(duration * v / total_count))
        if len(txt) == 0:
            txt = txt + str(k) + ' (' + percentage + '%, ' + final_duration + 's)'
        else:
            txt = txt + ' ' + str(k) + ' (' + percentage + '%, ' + final_duration + 's)'

    return txt

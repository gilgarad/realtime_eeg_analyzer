from flask import Flask, request, render_template
from flask_socketio import SocketIO
from os.path import join
import numpy as np

# app = Flask(__name__, template_folder='mdb_free_4.6.1')
app = Flask(__name__)
socketio = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'
# Bootstrap(app)
# app.config['BOOTSTRAP_SERVE_LOCAL'] = True #This turns file serving static
# api = Api(app)
realtime_emotion2 = list()

is_connection_request = [False]
hahaha = False

@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')
    # return render_template('index2.html', values=values, labels=labels)
    # return render_template('../mdb_free_4.6.1/index.html')


@socketio.on('update_data')
def test_message(data):
    # print('transmit!!')
    # data = {'username': 'isaacsim',
    #      'data': 'nothing'}
    data = {
        'eeg_mean': np.mean(data['eeg_realtime'], axis=0),
        'eeg_channels': data['eeg_realtime'].tolist(),
        'fun_accum': data['fun_accum'],
        'immersion_accum': data['immersion_accum'],
        'difficulty_accum': data['difficulty_accum'],
        'emotion_accum': data['emotion_accum'],
        'arousal_mean': np.mean(data['arousal_all'], axis=0).tolist(),
        'valence_mean': np.mean(data['valence_all'], axis=0).tolist(),
        'is_connected': data['is_connected'],
        'connection_status': data['connection_status'],
        'fun_mean': np.mean(data['fun_status'], axis=0).tolist(),
        'immersion_mean': np.mean(data['immersion_status'], axis=0).tolist(),
        'difficulty_mean': np.mean(data['difficulty_status'], axis=0).tolist(),
        'emotion_mean': np.mean(data['emotion_status'], axis=0).tolist()
    }
    socketio.emit('response', data, json=True, namespace='/update_data')


@socketio.on('connect_headset')
def connect_headset(message):
    # print('request in!!')
    # print(message)
    # realtime_emotion2[0].connect_headset()
    print(is_connection_request)
    is_connection_request[0] = not get_connection_request()


def set_realtime_emotion(rm):
    realtime_emotion2.append(rm)

def get_connection_request():
    return is_connection_request[0]

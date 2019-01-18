# from websocket import create_connection
import websocket
from realtime_eeg.emotiv_api import *
import ssl
import json
from os.path import join, abspath, dirname
config = json.load(open(join(join(dirname(__file__), '..'), 'config', 'account_config.json')))


class Emotiv:
    def __init__(self):
        self.is_run = True
        self.user_id = config['user_id']
        self.password = config['password']
        self.client_id = config['client_id']
        self.client_secret = config['client_secret']
        app_id = config['app_id']
        app_name = config['app_name']
        license_id = config['license_id']
        self.url = 'wss://emotivcortex.com:54321'
        self.ws = None
        self._auth = None
        self.is_connect = False


    def send_get_response(self, command):
        self.ws.send(json.dumps(command))
        res = json.loads(self.ws.recv())
        return res

    def connect_headset(self):
        print('Make a connection')
        # ws = create_connection(self.url, sslopt={"cert_reqs": ssl.CERT_NONE})
        self.ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
        self.ws.connect(self.url)

    def login(self):
        # 2. Check login status
        print('Check login status')
        # ws.send(json.dumps(getUserLogin()))
        # res = get_response(ws.recv())
        res = self.send_get_response(getUserLogin())
        if len(res['result']) == 0 or res['result'][0] != self.user_id:
            print('Currently not logged in. Trying to login.')
            print(res)
        else:
            print('Already logged in... logout then login again.')
            res = self.send_get_response(logout(self.user_id))

        res = self.send_get_response(login(self.user_id, self.password, self.client_id, self.client_secret))
        if 'result' in res and res['result'] == 'User ' + self.user_id + ' login successfully':
            print('User logged in successfully')
        else:
            print('User login failed.')
            print(res)

    def logout(self):
        res = self.send_get_response(logout(self.user_id))

    def authorize(self):
        # 3. authorize
        print('Trying to get authenticate')
        # ws.send(json.dumps(authorize(self.client_id, self.client_secret)))
        # res = get_response(ws.recv())
        res = self.send_get_response(authorize(self.client_id, self.client_secret))
        # print(res)
        if 'result' not in res:
            print('Authenticate failed')
            print(res)
        self._auth = res['result']['_auth']
        # print(res)
        # print(_auth)


    def get_license_info(self):
        # 4. get license info
        print('Get License Info')
        # ws.send(json.dumps(getLicenseInfo(_auth)))
        # res = get_response(ws.recv())
        res = self.send_get_response(getLicenseInfo(self._auth))
        # print(res)
        return res

    def query_headsets(self):
        # 5. query headsets
        print('Query Headsets')
        # ws.send(json.dumps(queryHeadsets()))
        # res = get_response(ws.recv())
        res = self.send_get_response(queryHeadsets())
        if 'result' in res and len(res['result']) == 0:
            print('No device connected')
        # print(res)
        return res

    def close_old_sessions(self):
        # 6. query sessions
        print('Query Sessions')
        # ws.send(json.dumps(querySessions(_auth)))
        # res = get_response(ws.recv())
        res = self.send_get_response(querySessions(self._auth))
        # print(res)
        print('Number of sessions opened:', len(res['result']))
        for s in res['result']:
            if s['status'] != 'closed':
                print('Closing old session id:', s['id'])
                # ws.send(json.dumps(updateSession(_auth, s['id'])))
                # res = get_response(ws.recv())
                res = self.send_get_response(updateSession(self._auth, s['id']))

        return res

    def create_session(self):
        # 7. create session
        print('Create a new Session')
        # ws.send(json.dumps(createSession(_auth)))
        # res = get_response(ws.recv())
        res = self.send_get_response(createSession(self._auth))
        print(res)

        # only 'result' or 'error' should be used for determine if it can proceed or not
        # print(res)
        # open
        # {'id': 1, 'jsonrpc': '2.0', 'result': {'appId': 'com.igsinc.eeg_emotion', 'headset': {'connectedBy': 'bluetooth', 'dongle': '0', 'firmware': '625', 'id': 'EPOCPLUS-3b9ae5f4', 'label': '', 'motionSensors': ['GYROX', 'GYROY', 'GYROZ', 'ACCX', 'ACCY', 'ACCZ', 'MAGX', 'MAGY', 'MAGZ'], 'sensors': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'], 'settings': {'eegRate': 128, 'eegRes': 16, 'memsRate': 0, 'memsRes': 16, 'mode': 'EPOCPLUS'}, 'status': 'connected'}, 'id': 'e1d3f85a-1e3f-47b7-902e-f882dbf74f6f', 'license': '34a46e15-9018-4d51-83f1-b92db2b30e05', 'logs': {'recordInfos': []}, 'markers': [], 'owner': 'igsinc', 'profile': '', 'project': '', 'recording': None, 'started': '2018-11-08T14:35:55.230089+09:00', 'status': 'opened', 'stopped': '', 'streams': None, 'subject': 0, 'tags': [], 'title': ''}}
        # active
        # {'id': 1, 'jsonrpc': '2.0', 'result': {'appId': 'com.igsinc.eeg_emotion', 'headset': {'connectedBy': 'bluetooth', 'dongle': '0', 'firmware': '625', 'id': 'EPOCPLUS-3b9ae5f4', 'label': '', 'motionSensors': ['GYROX', 'GYROY', 'GYROZ', 'ACCX', 'ACCY', 'ACCZ', 'MAGX', 'MAGY', 'MAGZ'], 'sensors': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'], 'settings': {'eegRate': 128, 'eegRes': 16, 'memsRate': 0, 'memsRes': 16, 'mode': 'EPOCPLUS'}, 'status': 'connected'}, 'id': '63025c0b-0138-4c48-84b6-739e2112e047', 'license': '34a46e15-9018-4d51-83f1-b92db2b30e05', 'logs': {'recordInfos': []}, 'markers': [], 'owner': 'igsinc', 'profile': '', 'project': '', 'recording': False, 'started': '2018-11-08T14:46:39.434741+09:00', 'status': 'activated', 'stopped': '', 'streams': None, 'subject': 0, 'tags': [], 'title': ''}}

    def subscribe(self):
        # 8. subscribe
        self.ws.send(json.dumps(subscribe(self._auth)))
        self.is_connect = True
        # print(res)

    def unsubscribe(self):
        self.ws.send(json.dumps(unsubscribe(self._auth)))
        self.is_connect = False

    def retrieve_packet(self):
        return json.loads(self.ws.recv())

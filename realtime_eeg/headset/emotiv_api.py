""" emotiv_api.py: All the api json implemented by the emotiv instruction
website: https://emotiv.github.io/cortex-docs/#data-types
"""

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"


# Mandatory Methods by Cortex API
def logout(user_id: str):
    """ Logout

    :param user_id:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "logout",
        "params": {
            "username": user_id
        },
        "id": 1
    }


def login(user_id: str, password: str, client_id: str, client_secret: str):
    """ Login

    :param user_id:
    :param password:
    :param client_id:
    :param client_secret:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "login",
        "params": {
            "username": user_id,
            "password": password,
            "client_id": client_id,
            "client_secret": client_secret
        },
        "id": 1
    }


def getUserLogin():
    """ Get all users logged in to Cortex

    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "getUserLogin",
        "id": 1
    }


def authorize(client_id: str, client_secret: str):
    """ Authenticate a user

    :param client_id:
    :param client_secret:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "authorize",
        "params": {
            "client_id": client_id,
            "client_secret": client_secret,
            # "license": license_id,
            # "debit": 0
        },
        "id": 1
    }


def acceptLicense(_auth: str):
    """ Accept license in order to use Cortex

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "acceptLicense",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def getLicenseInfo(_auth: str):
    """ Get license information of a user

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "getLicenseInfo",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def queryHeadsets():
    """ Shows the detailed list of headsets connected

    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "queryHeadsets",
        "params": {},
        "id": 1
    }


def createSession(_auth: str):
    """ Creates a new session

    :param _auth:
    :return:
    """
    # active, close, startRecord, stopRecord, addTags, removeTags
    return {
        "jsonrpc": "2.0",
        "method": "createSession",
        "params": {
            "_auth": _auth,
            # "status": "open"
            # "status": "close"
            "status": "active"
        },
        "id": 1
    }


def updateSession(_auth: str, sess_id: str):
    """ Update the session with new information

    :param _auth:
    :param sess_id:
    :return:
    """
    # status: active, close, startRecord, stopRecord, addTags, removeTags
    return {
        "jsonrpc": "2.0",
        "method": "updateSession",
        "params": {
            "_auth": _auth,
            "session": sess_id,
            "status": "close"
        },
        "id": 1
    }


def querySessions(_auth: str):
    """ Query the list of all sessions

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "querySessions",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def subscribe(_auth: str):
    """ Subscribe to the streams of dev, eeg, pow, met

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "subscribe",
        "params": {
            "_auth": _auth,
            "streams": [
                "dev", # connection status to the cortex
                "eeg", # eeg rawdata
                "pow", # theta alpha betaL betaH gamma
                "met" # emotiv's analyzed result of
                        # interest, stress, relaxation, excitement, engagement, long term excitement, focus
            ]
        },
        "id": 1
    }


def updateNote(_auth: str):
    """ Edit note when recording is on a session

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "updateNote",
        "params": {
            "_auth": _auth,
            "session": "abcd",
            "note": "new note",
            "record": "1234"
        },
        "id": 1
    }


def unsubscribe(_auth: str):
    """ Stop subscribe to the streams of dev, eeg, pow, met

    :param _auth:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "unsubscribe",
        "params": {
            "_auth": _auth,
            "streams": [
                "eeg",
                "dev",
                "pow",
                "met"
            ]
        },
        "id": 1
    }


def injectMarker(_auth: str, sess: str):
    """ Injects a marker into the data stream for a headset

    :param _auth:
    :param sess:
    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "injectMarker",
        "params": {
            "_auth": _auth,
            "session": sess,
            "label": "test1",
            "value": "record-1",
            "port": "USB",
            "time": 123456789
        },
        "id": 1
    }


def getDetectionInfo():
    """ This request return all useful informations for set up training Mental Command and Facial Expression

    :return:
    """
    return {
        "jsonrpc": "2.0",
        "method": "getDetectionInfo",
        "params": {
            "detection": "mentalCommand"
        },
        "id": 1
    }



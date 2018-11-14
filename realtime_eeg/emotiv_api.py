# Mandatory Methods by Cortex API
def logout(user_id):
    return {
        "jsonrpc": "2.0",
        "method": "logout",
        "params": {
            "username": user_id
        },
        "id": 1
    }


def login(user_id, password, client_id, client_secret):
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
    return {
        "jsonrpc": "2.0",
        "method": "getUserLogin",
        "id": 1
    }


def authorize(client_id, client_secret):
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


def acceptLicense(_auth):
    return {
        "jsonrpc": "2.0",
        "method": "acceptLicense",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def getLicenseInfo(_auth):
    return {
        "jsonrpc": "2.0",
        "method": "getLicenseInfo",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def queryHeadsets():
    return {
        "jsonrpc": "2.0",
        "method": "queryHeadsets",
        "params": {},
        "id": 1
    }


def createSession(_auth):
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


def updateSession(_auth, sess_id):
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


def querySessions(_auth):
    return {
        "jsonrpc": "2.0",
        "method": "querySessions",
        "params": {
            "_auth": _auth
        },
        "id": 1
    }


def subscribe(_auth):
    return {
        "jsonrpc": "2.0",
        "method": "subscribe",
        "params": {
            "_auth": _auth,
            "streams": [
                "eeg"
            ]
        },
        "id": 1
    }


def updateNote(_auth):
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


def unsubscribe(_auth):
    return {
        "jsonrpc": "2.0",
        "method": "unsubscribe",
        "params": {
            "_auth": _auth,
            "streams": [
                "eeg"
            ]
        },
        "id": 1
    }


def injuectMarker(_auth, sess):
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
    return {
        "jsonrpc": "2.0",
        "method": "getDetectionInfo",
        "params": {
            "detection": "mentalCommand"
        },
        "id": 1
    }



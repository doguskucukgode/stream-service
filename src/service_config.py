import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)


# Server configs
service = {
    "host" : "0.0.0.0",
    "port" : "54443",
    "ZMQ_URL_CR_CL" : "tcp://localhost:54321",
    "ZMQ_URL_FACE" : "tcp://localhost:54444"
}

# Available actions
actions = {
    "ACTION_START" : 0,
    "ACTION_STOP" : 1,
    "ACTION_CHECK" : 2
}

# Stream-related configs
stream = {
    "TYPE_CAR_CLASSIFICATION" : 0,
    "TYPE_FACE_DETECTION" : 1,
    "INTERVAL" : 24,
    "COPY_COUNT" : 5,
    "RECONNECT_TIME_OUT" : 5,
    "RECONNECT_TRY_COUNT" : 5
}

# Configs related to check stream status
stream_stat = {
    "auth-user" : "dogus",
    "auth-pass" : "ares2728",
    "url" : "http://localhost:8087/v2/servers/_defaultServer_/vhosts/_defaultVHost_/applications/live/instances/_definst_",
    "headers" : {
        'Accept': 'application/json; charset=utf-8',
    }
}

# Available actions
actions = {
    "ACTION_START" : 0,
    "ACTION_STOP" : 1,
    "ACTION_CHECK" : 2
}

# Stream-related configs
stream = {
    "TYPE_CAR_CLASSIFICATION" : 0,
    "TYPE_FACE_DETECTION" : 1,
    "INTERVAL" : 24,
    "COPY_COUNT" : 5,
    "RECONNECT_TIME_OUT" : 5,
    "RECONNECT_TRY_COUNT" : 5
}

# Configs related to check stream status
stream_stat = {
    "auth-user" : "dogus",
    "auth-pass" : "ares2728",
    "url" : "http://localhost:8087/v2/servers/_defaultServer_/vhosts/_defaultVHost_/applications/live/instances/_definst_",
    "headers" : {
        'Accept': 'application/json; charset=utf-8',
    }
}

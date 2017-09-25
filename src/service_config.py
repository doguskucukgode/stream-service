import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)


# Server configs
service = {
    "host" : "0.0.0.0",
    "port" : "55555",
    "ZMQ_URL_CR_CL" : "tcp://192.168.0.94:54321",
    "ZMQ_URL_FACE" : "tcp://192.168.0.94:54444"
}

# Available actions
actions = {
    "ACTION_START" : 0,
    "ACTION_STOP" : 1
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

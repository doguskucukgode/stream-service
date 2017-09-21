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

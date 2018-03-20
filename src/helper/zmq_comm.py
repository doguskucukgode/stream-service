# Everything related to ZMQ communication is implemented here

import cv2
import zmq
import base64
import numpy as np


def init_server(context, address):
    """Given a context and an address, inits a server socket and returns it."""
    try:
        socket = context.socket(zmq.REP)
        socket.bind(address)
        return socket
    except Exception as e:
        message = "Could not initialize the server: "
        print(message + str(e))
        raise Exception(message)


def init_client(context, address):
    """Given a context and an address, inits a client socket and returns it."""
    try:
        socket = context.socket(zmq.REQ)
        socket.connect(address)
        return socket
    except Exception as e:
        message = "Could not initialize the client: "
        print(message + str(e))
        raise Exception(message)


def decode_request(request):
    """Decodes an incoming base64 request and extracts the image from it."""
    try:
        img = base64.b64decode(request)
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        message = "Could not decode the received request: "
        print(message + str(e))
        raise Exception(message)


def get_tcp_address(host, port):
    """Given host ip and port, returns a TCP address."""
    return "tcp://" + host + ":" + port

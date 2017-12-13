# External imports
import os
import re
import sys
import cv2
import zmq
import uuid
import json
import base64
import itertools
import numpy as np

import keras
from keras import backend as K
from keras.models import Model, load_model

# Internal imports
import plate_conf

os.environ["CUDA_VISIBLE_DEVICES"]="1"
current_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

tf_config = K.tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = K.tf.Session(config=tf_config)
K.set_session(sess)

alphabet = {
    0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
    10: 'A',  11: 'B',  12: 'C',  13: 'D',  14: 'E',  15: 'F',  16: 'G',  17: 'H',  18: 'I',
    19: 'J',  20: 'K',  21: 'L',  22: 'M',  23: 'N',  24: 'O',  25: 'P',  26: 'R',  27: 'S',
    28: 'T',  29: 'U',  30: 'V',  31: 'Y',  32: 'Z',  33: ' '
}


def init_server(address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(address)
        return socket
    except Exception as e:
        message = "Could not initialize the server."
        print (message + str(e))
        raise Exception(message)


# Extracts the image from the received request
def decode_request(request):
    try:
        img = base64.b64decode(request)
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        message = "Could not decode the received request."
        print(message + str(e))
        raise Exception(message)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))

        out_str_wo_gb = ''
        for c in out_best:
            if c < len(alphabet):
                out_str_wo_gb = out_str_wo_gb + alphabet[c] + '.'
        # print(out_str_wo_gb)

        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(alphabet):
                outstr += alphabet[c]
        ret.append(outstr)
    return ret


def almost_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    else:
        count = 0
        for a, b in zip(w1, w2):
            if a != b :
                count += 1
            if count == 2:
                return False
        else:
            return True


def detect_plates(plate_classifier, iteration_inc, strictness, img):
    plates = plate_classifier.detectMultiScale(img, iteration_inc, strictness)
    # print("Found plates: ", plates)
    return plates


def handle_requests(socket, plate_recognizer):
    # Load recognition related configs
    image_width = plate_conf.recognition["image_width"]
    image_height = plate_conf.recognition["image_height"]

    # Compile regex that matches with invalid TR plates
    invalid_tr_plate_regex = plate_conf.recognition["invalid_tr_plate_regex"]
    invalid_plate_pattern = re.compile(invalid_tr_plate_regex)
    print("Plate recognition is started on: ", tcp_address)

    net_inp = plate_recognizer.get_layer(name='the_input').input
    net_out = plate_recognizer.get_layer(name='softmax').output

    while True:
        result_dict = {}
        message = "OK"
        found_plate = ""
        topleft = {}
        bottomright = {}
        coord_info = {}

        try:
            # Get image from socket and perform detection
            request = socket.recv()
            image = decode_request(request)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (image_width, image_height))
            gray = gray.astype(np.float32)
            gray /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            gray = gray.T   # (128, 32)
            gray = np.expand_dims(gray, -1) # (128, 32, 1)

            # Feeding a single image at a time: (1, image_width, image_height, 1)
            # Otherwise X_data should be of shape: (n, image_width, image_height, 1)
            X_data = np.ones([1, image_width, image_height, 1])
            X_data[0] = gray
            net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
            pred_texts = decode_batch(net_out_value)
            if len(pred_texts) > 0:
                found_plate = pred_texts[0]

        except Exception as e:
            message = str(e)

        all_info = {}
        all_info["plate"] = found_plate
        all_info["coords"] = coord_info
        result_dict["result"] = [all_info]
        result_dict["message"] = message
        # print(result_dict)
        socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    plate_recognizer = None

    # Load plate recognizer once
    model_path = plate_conf.recognition['model_path']
    try:
        print("Loading model: ", model_path)
        plate_recognizer = load_model(model_path, compile=False)
    except Exception as e:
        exception_info = str(e)
        print("Error while loading plate recognizer: " + exception_info)
        sys.exit()

    try:
        host = plate_conf.recognizer_server['host']
        port = plate_conf.recognizer_server['port']
        tcp_address = plate_conf.get_tcp_address(host, port)
        socket = init_server(tcp_address)
        handle_requests(socket, plate_recognizer)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()

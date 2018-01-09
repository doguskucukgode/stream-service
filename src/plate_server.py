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
import zmq_comm
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


def detect_plates(plate_classifier, iteration_inc, strictness, img):
    plates = plate_classifier.detectMultiScale(img, iteration_inc, strictness)
    # print("Found plates: ", plates)
    return plates


def handle_requests(socket, plate_detector, plate_recognizer):
    iteration_inc = plate_conf.detection["detection_iteration_increase"]
    strictness = plate_conf.detection["detection_strictness"]

    # Load recognition related configs
    required_image_width = plate_conf.recognition["image_width"]
    required_image_height = plate_conf.recognition["image_height"]

    # Compile regex that matches with invalid TR plates
    invalid_tr_plate_regex = plate_conf.recognition["invalid_tr_plate_regex"]
    invalid_plate_pattern = re.compile(invalid_tr_plate_regex)
    net_inp = plate_recognizer.get_layer(name='the_input').input
    net_out = plate_recognizer.get_layer(name='softmax').output
    print("Plate server is started on: ", tcp_address)

    while True:
        result_dict = {}
        message = "OK"
        all_detected_plates = []
        found_plate = ""

        try:
            # Get image from socket and perform detection
            request = socket.recv()
            image = zmq_comm.decode_request(request)
            #TODO: check len of shape it must be three
            # Do not attempt manipulating image if it is already grayscale
            image_h, image_w, dim = image.shape
            if dim != 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            plate_coords = detect_plates(plate_detector, iteration_inc, strictness, image)
            for (x, y, w, h) in plate_coords:
                topleft = {}
                bottomright = {}
                coord_info = {}

                topleft["x"] = int(x)
                topleft["y"] = int(y)
                bottomright["x"] = int(x + w)
                bottomright["y"] = int(y + h)
                coord_info["topleft"] = topleft
                coord_info["bottomright"] = bottomright

                detected = {}
                detected["coords"] = coord_info
                detected["area"] = int(w * h)
                all_detected_plates.append(detected)

            all_detected_plates = sorted(all_detected_plates, key=lambda detected: detected["area"], reverse=True)
            if len(all_detected_plates) > 0:
                # Feeding a single image at a time: (1, image_width, image_height, 1)
                # Otherwise X_data should be of shape: (n, image_width, image_height, 1)
                X_data = np.ones([len(all_detected_plates), required_image_width, required_image_height, 1])

                for index, detected_plate in enumerate(all_detected_plates):
                    coord_info = detected_plate['coords']
                    # If coordinate info is empty, return empty plate since we could not found any plates
                    if not coord_info:
                        print("Coord info is empty, could not find any plates..")
                        break

                    # Crop the plate out of the car image
                    topleft_x = int(coord_info['topleft']['x'])
                    topleft_y = int(coord_info['topleft']['y'])
                    bottomright_x = int(coord_info['bottomright']['x'])
                    bottomright_y = int(coord_info['bottomright']['y'])
                    width = int(bottomright_x - topleft_x)
                    height = int(bottomright_y - topleft_y)
                    margin_width = int(height / 2)
                    margin_height = int(height / 4)

                    # Add margins
                    topleft_x = max(0, topleft_x - margin_width)
                    topleft_y = max(0, topleft_y - margin_height)
                    bottomright_x = min(image_w, bottomright_x + margin_width)
                    bottomright_y = min(image_h, bottomright_y + margin_height)
                    # Crop the detected plate
                    cropped_plate_img = image[topleft_y:bottomright_y, topleft_x:bottomright_x]
                    # Recognize the cropped plate image
                    cropped_plate_img = cv2.resize(cropped_plate_img, (required_image_width, required_image_height))
                    cropped_plate_img = cropped_plate_img.astype(np.float32)
                    cropped_plate_img /= 255

                    # width and height are backwards from typical Keras convention
                    # because width is the time dimension when it gets fed into the RNN
                    cropped_plate_img = cropped_plate_img.T   # (128, 32)
                    cropped_plate_img = np.expand_dims(cropped_plate_img, -1) # (128, 32, 1)

                    # Populate X_data with cropped and processed plate regions
                    X_data[index] = cropped_plate_img

                net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
                pred_texts = decode_batch(net_out_value)
                filtered_candidates = []
                for plate_text in pred_texts:
                    # If our regex does not match with a plate, then it is a good candidate
                    if not invalid_plate_pattern.search(plate_text):
                        filtered_candidates.append(plate_text)

                if len(filtered_candidates) > 0:
                    found_plate = filtered_candidates[0]
        except Exception as e:
            message = str(e)

        result_dict["result"] = found_plate
        result_dict["message"] = message
        # print(result_dict)
        socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    plate_detector = None
    plate_recognizer = None

    # Load configs and plate detector once
    classifier_path = plate_conf.detection['classifier_path']
    plate_detector = cv2.CascadeClassifier(classifier_path)
    if plate_detector.empty():
        print("Error while loading plate detector at given path: " + classifier_path)
        sys.exit()

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
        host = plate_conf.plate_server['host']
        port = plate_conf.plate_server['port']
        tcp_address = zmq_comm.get_tcp_address(host, port)
        ctx = zmq.Context(io_threads=1)
        socket = zmq_comm.init_server(ctx, tcp_address)
        handle_requests(socket, plate_detector, plate_recognizer)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()

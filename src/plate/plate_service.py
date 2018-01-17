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
from service import Service
import helper.zmq_comm as zmq_comm

class PlateService(Service):

    def __init__(self, machine=None):
        self.plate_detector = None
        self.plate_recognizer = None
        self.alphabet = {
            0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
            10: 'A',  11: 'B',  12: 'C',  13: 'D',  14: 'E',  15: 'F',  16: 'G',  17: 'H',  18: 'I',
            19: 'J',  20: 'K',  21: 'L',  22: 'M',  23: 'N',  24: 'O',  25: 'P',  26: 'R',  27: 'S',
            28: 'T',  29: 'U',  30: 'V',  31: 'Y',  32: 'Z',  33: ' '
        }
        super().__init__(machine)

        # os.environ["CUDA_VISIBLE_DEVICES"]="1"
        self.tf_config = K.tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = K.tf.Session(config=self.tf_config)
        K.set_session(self.sess)
        self.load()
        self.handle_requests()

    def get_server_configs(self):
        return self.configs.plate_server["host"], self.configs.plate_server["port"]

    def load(self):
        # Load configs and plate detector once
        classifier_path = self.configs.detection['classifier_path']
        try:
            self.plate_detector = cv2.CascadeClassifier(classifier_path)
            if self.plate_detector is None or self.plate_detector.empty():
                print("Error while loading plate detector at given path: " + classifier_path)
                sys.exit()
        except Exception as e:
            print("Error while loading plate detector: " + str(e))
            sys.exit(status=1)

        # Load plate recognizer once
        model_path = self.configs.recognition['model_path']
        try:
            print("Loading model: ", model_path)
            self.plate_recognizer = load_model(model_path, compile=False)
        except Exception as e:
            print("Error while loading plate recognizer: " + str(e))
            sys.exit(status=1)

    # For a real OCR application, this should be beam search with a dictionary
    # and language model.  For this example, best path is sufficient.
    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))

            out_str_wo_gb = ''
            for c in out_best:
                if c < len(self.alphabet):
                    out_str_wo_gb = out_str_wo_gb + self.alphabet[c] + '.'
            # print(out_str_wo_gb)

            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(self.alphabet):
                    outstr += self.alphabet[c]
            ret.append(outstr)
        return ret

    def handle_requests(self):
        # Load detection related configs
        iteration_inc = self.configs.detection["detection_iteration_increase"]
        strictness = self.configs.detection["detection_strictness"]
        # Load recognition related configs
        required_image_width = self.configs.recognition["image_width"]
        required_image_height = self.configs.recognition["image_height"]
        # Compile regex that matches with invalid TR plates
        invalid_tr_plate_regex = self.configs.recognition["invalid_tr_plate_regex"]
        invalid_plate_pattern = re.compile(invalid_tr_plate_regex)
        net_inp = self.plate_recognizer.get_layer(name='the_input').input
        net_out = self.plate_recognizer.get_layer(name='softmax').output
        print("Plate server is started on: ", self.address)

        while True:
            result_dict = {}
            message = "OK"
            all_detected_plates = []
            found_plate = ""

            try:
                # Get image from socket and perform detection
                request = self.socket.recv()
                image = zmq_comm.decode_request(request)
                #TODO: check len of shape it must be three
                # Do not attempt manipulating image if it is already grayscale
                image_h, image_w, dim = image.shape
                if dim != 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                plate_coords = self.plate_detector.detectMultiScale(image, iteration_inc, strictness)
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

                    net_out_value = self.sess.run(net_out, feed_dict={net_inp:X_data})
                    pred_texts = self.decode_batch(net_out_value)
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
            self.socket.send_json(result_dict)

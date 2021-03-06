# Plate detection service: it uses OpenCV's CascadeClassifier

# External imports
import os
import sys
import cv2
import zmq
import base64
import numpy as np

# Internal imports
module_folder = os.path.dirname(os.path.realpath(__file__))
source_folder = os.path.dirname(module_folder)
base_folder = os.path.dirname(source_folder)
model_folder = base_folder + "/model"
sys.path.insert(0, source_folder)
from conf.plate_conf import PlateConfig
import helper.zmq_comm as zmq_comm


def detect_plates(plate_classifier, iteration_inc, strictness, img):
    plates = plate_classifier.detectMultiScale(img, iteration_inc, strictness)
    # print("Found plates: ", plates)
    return plates


def handle_requests(socket, plate_detector):
    iteration_inc = PlateConfig.detection["detection_iteration_increase"]
    strictness = PlateConfig.detection["detection_strictness"]
    print("Plate detection is started on: ", tcp_address)

    while True:
        result_dict = {}
        message = "OK"
        all_detected_plates = []

        try:
            # Get image from socket and perform detection
            request = socket.recv()
            image = zmq_comm.decode_request(request)
            #TODO: check len of shape it must be three
            # Do not attempt manipulating image if it is already grayscale
            if image.shape[2] != 1:
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

        except Exception as e:
            message = str(e)


        all_detected_plates = sorted(all_detected_plates, key=lambda detected: detected["area"], reverse=True)
        result_dict["result"] = all_detected_plates
        result_dict["message"] = message
        # print(result_dict)
        socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    plate_detector = None

    # Load configs and plate detector once
    classifier_path = PlateConfig.detection['classifier_path']
    plate_detector = cv2.CascadeClassifier(classifier_path)
    if plate_detector.empty():
        print("Error while loading plate detector at given path: " + classifier_path)
        sys.exit()

    try:
        host = PlateConfig.detector_server['host']
        port = PlateConfig.detector_server['port']
        tcp_address = zmq_comm.get_tcp_address(host, port)
        ctx = zmq.Context(io_threads=1)
        socket = zmq_comm.init_server(ctx, tcp_address)
        handle_requests(socket, plate_detector)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()

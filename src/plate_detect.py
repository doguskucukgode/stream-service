# External imports
import os
import sys
import cv2
import zmq
import base64
import numpy as np

# Internal imports
import plate_conf


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


def detect_plates(plate_classifier, iteration_inc, strictness, img):
    plates = plate_classifier.detectMultiScale(img, iteration_inc, strictness)
    # print("Found plates: ", plates)
    return plates


def handle_requests(socket, plate_detector):
    iteration_inc = plate_conf.detection["detection_iteration_increase"]
    strictness = plate_conf.detection["detection_strictness"]
    print("Plate detection is started on: ", tcp_address)

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
            plate_coords = detect_plates(plate_detector, iteration_inc, strictness, gray)
            if len(plate_coords) > 0:
                for (x, y, w, h) in plate_coords:
                    topleft["x"] = int(x)
                    topleft["y"] = int(y)
                    bottomright["x"] = int(x + w)
                    bottomright["y"] = int(y + h)
                    coord_info["topleft"] = topleft
                    coord_info["bottomright"] = bottomright
                    found_plate = "FOUND"
                    # Just sends the first result for the time being
                    break

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
    plate_detector = None

    # Load configs and plate detector once
    classifier_path = plate_conf.detection['classifier_path']
    plate_detector = cv2.CascadeClassifier(classifier_path)
    if plate_detector.empty():
        print("Error while loading plate detector at given path: " + classifier_path)
        sys.exit()

    try:
        host = plate_conf.detector_server['host']
        port = plate_conf.detector_server['port']
        tcp_address = plate_conf.get_tcp_address(host, port)
        socket = init_server(tcp_address)
        handle_requests(socket, plate_detector)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()

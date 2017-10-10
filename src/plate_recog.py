# External imports
import os
import re
import sys
import cv2
import zmq
import json
import base64
import numpy as np
from openalpr import Alpr

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
        # OpenALPR requires byte array
        return bytes(img)
    except Exception as e:
        message = "Could not decode the received request."
        print(message + str(e))
        raise Exception(message)


def handle_requests(socket, alpr):
    # Load configs and Alpr() once
    country = plate_conf.recognition['country']
    region = plate_conf.recognition['region']
    openalpr_conf_dir = plate_conf.recognition['openalpr_conf_dir']
    openalpr_runtime_data_dir = plate_conf.recognition['openalpr_runtime_data_dir']
    top_n = plate_conf.recognition['top_n']

    # Compile regex that matches with invalid TR plates
    invalid_tr_plate_regex = plate_conf.recognition["invalid_tr_plate_regex"]
    invalid_plate_pattern = re.compile(invalid_tr_plate_regex)

    alpr = Alpr(country, openalpr_conf_dir, openalpr_runtime_data_dir)
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        return

    alpr.set_top_n(top_n)
    alpr.set_default_region(region)
    print("Plate recognition is started on: ", tcp_address)

    while True:
        result_dict = {}
        message = "OK"
        found_plate = ""

        try:
            # Get image from socket
            request = socket.recv()
            image = decode_request(request)
            # results = alpr.recognize_file("/home/taylan/gitFolder/stream-service/received_by_plate.jpg")
            results = alpr.recognize_array(image)
            # print("Results: ", results)

            filtered_candidates = []
            for i, plate in enumerate(results['results']):
                for candidate in plate['candidates']:
                    print(candidate['plate'])
                    # If our regex does not match with a plate, then it is a good candidate
                    if not invalid_plate_pattern.search(candidate['plate']):
                        filtered_candidates.append(candidate['plate'])
                # WARNING: It is assumed that there is only a single plate in the given image
                # Hence, we break after the first plate, even if there are more plates
                break

            # print(filtered_candidates)
            if len(filtered_candidates) > 0:
                found_plate = filtered_candidates[0]

        except Exception as e:
            message = str(e)

        result_dict["result"] = found_plate
        result_dict["message"] = message
        socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    alpr = None
    try:
        tcp_address = plate_conf.get_tcp_address(plate_conf.server["host"], plate_conf.server["port"])
        socket = init_server(tcp_address)
        handle_requests(socket, alpr)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()
        if alpr is not None:
            # Call when completely done to release memory
            alpr.unload()
            print("ALPR unloaded properly.")

# External imports
import os
import re
import sys
import cv2
import zmq
import json
import uuid
import base64

# Internal imports
module_folder = os.path.dirname(os.path.realpath(__file__))
source_folder = os.path.dirname(module_folder)
base_folder = os.path.dirname(source_folder)
model_folder = base_folder + "/model"
sys.path.insert(0, source_folder)
from conf.plate_conf import PlateConfig
import helper.zmq_comm as zmq_comm


def read_image(path):
    try:
        img = cv2.imread(path)
        cv_encoded_img = cv2.imencode(".jpg", img)[1]
        encoded_img = base64.b64encode(cv_encoded_img)
        return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def take_action(image, message, image_path, out_path):
    tr_plate_regex = PlateConfig.recognition["tr_plate_regex"]
    plate_pattern = re.compile(tr_plate_regex)
    results = json.loads(message.decode("utf-8"))['result']
    try:
        for res in results:
            plate = res["plate"]
            coord_info = res['coords']
            cropped_plate_img = image
            if coord_info:
                topleft = coord_info['topleft']
                bottomright = coord_info['bottomright']

                width = int(bottomright['x'] - topleft['x'])
                height = int(bottomright['y'] - topleft['y'])
                margin_width = int(height / 2)
                margin_height = int(height / 4)

                topleft_x = topleft['x'] - margin_width
                topleft_y = topleft['y'] - margin_height
                bottomright_x = bottomright['x'] + margin_width
                bottomright_y = bottomright['y'] + margin_height

                cropped_plate_img = image[topleft_y:bottomright_y, topleft_x:bottomright_x]
                if plate == '':
                    plate = "NOPLATE"
                cv2.imwrite(out_path + "/" + str(plate) + '_' + str(uuid.uuid4()) + '.jpg', cropped_plate_img)
                print("Cropped plate image is written under: " + out_path)

            if plate:
                if plate_pattern.search(plate):
                    cv2.imwrite(out_path + "/" + str(plate) + '_' + str(uuid.uuid4()) + '.jpg', cropped_plate_img)
                else:
                    print("Did not save this: ", plate)
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    host = PlateConfig.plate_server['host']
    port = PlateConfig.plate_server['port']
    address = zmq_comm.get_tcp_address(host, port)
    ctx = zmq.Context(io_threads=1)
    socket = zmq_comm.init_client(ctx, address)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    unprocessed_imgs_path = "/home/taylan/local/plate_data/cropped"
    output_path = "/home/taylan/local/plate_data/plate_test_regex"

    # Recursively reads images under a given directory path, queries the plates in images one by one
    image_paths = []
    for dirName, subdirList, fileList in os.walk(unprocessed_imgs_path):
        print(dirName)
        for fname in fileList:
            p = dirName + '/' + fname
            if os.path.isfile(p):
                image_paths.append(p)

    print ("Sending requests..")
    for index, image_path in enumerate(image_paths):
        try:
            encoded_img = read_image(image_path)
            socket.send(encoded_img)
            message = socket.recv()
            print("Sent: ", image_path)
            print ("Received reply: ", message)
            try:
                image = cv2.imread(image_path, 1)
                take_action(image, message, image_path, output_path)
            except Exception as e:
                print ("Could not annotate the given image.")
                print(e)
        except Exception as e:
            print("Could not send the image due to: ", str(e))

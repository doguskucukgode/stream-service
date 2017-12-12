import os
import cv2
import zmq
import json
import uuid
import base64

import plate_conf


def init_client(address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(address)
        return socket
    except Exception as e:
        print ("Could not initialize the client: " + str(e))


def read_image(path):
    try:
        img = cv2.imread(path)
        cv_encoded_img = cv2.imencode(".jpg", img)[1]
        encoded_img = base64.b64encode(cv_encoded_img)
        return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def take_action(image, message, image_path, out_path):
    results = json.loads(message.decode("utf-8"))['result']
    try:
        for res in results:
            plate = res['plate']
            coord_info = res['coords']

            if not coord_info:
                print("Could not detect any plates, deleting this image..")
                # os.remove(image_path)
            else:
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
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    host = plate_conf.recognizer_server['host']
    port = plate_conf.recognizer_server['port']
    address = plate_conf.get_tcp_address(host, port)
    socket = init_client(address)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    unprocessed_imgs_path = "/home/taylan/gitFolder/plate-deep-ocr-v2/plate_data_duz"
    output_path = "/home/taylan/Desktop/car_images/recog_results/"

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

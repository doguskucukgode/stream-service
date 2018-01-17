# External imports
import os
import cv2
import zmq
import json
import time
import base64

# Internal imports
module_folder = os.path.dirname(os.path.realpath(__file__))
source_folder = os.path.dirname(module_folder)
base_folder = os.path.dirname(source_folder)
model_folder = base_folder + "/model"
sys.path.insert(0, source_folder)
from conf.face_conf import FaceConfig
import helper.zmq_comm as zmq_comm

def init_client(address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(address)
        return socket
    except Exception as e:
        print ("Could not initialize the client: " + str(e))


def read_image_base64(path):
    try:
        with open(path, mode="rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())
            return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def read_image_new(path):
    try:
        img = cv2.imread(path)
        cv_encoded_img = cv2.imencode(".jpg", img)[1]
        encoded_img = base64.b64encode(cv_encoded_img)
        return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def evaluate_face(image_class, message, out_path):
    results = message['result']
    for res in results:
        topleft = res['topleft']
        bottomright = res['bottomright']
        name = res['name']
        if image_class == name:
            return True
        else:
            return False


if __name__ == '__main__':
    # Set server info, you may use configs given in configurations
    host = FaceConfig.server['host']
    port = FaceConfig.server['port']

    address = zmq_comm.get_tcp_address(host, port)
    socket = init_client(address)

    # Other stuff
    current_dir = os.path.dirname(os.path.realpath(__file__))
    eval_folder = "/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI/"
    image_list = os.listdir(eval_folder)
    image_list = [eval_folder + i for i in image_list]
    total_num_of_images = len(image_list)
    total_num_of_frontal = total_num_of_images - 400
    correctly_recognized = 0
    correctly_recognized_frontal = 0

    start = time.time()
    for im_path in image_list:
        print("Querying: ", im_path)
        im_name = im_path.split('/')[-1]
        parts = im_name.split('-')
        im_class = parts[0]
        im_index = parts[1]
        img = cv2.imread(im_path)
        encoded_img = read_image_new(im_path)
        socket.send(encoded_img)
        message = socket.recv_json()
        try:
            evaluation = evaluate_face(im_class, message, current_dir)
            if evaluation is True:
                correctly_recognized += 1
                if im_index != "01" or im_index != "10":
                    correctly_recognized_frontal += 1
            else:
                cv2.imwrite("/home/taylan/local/face-recognition/datasets/FEI_Brazil/wrong_classifications/wrong_" + str(im_class) + "-" + str(im_index), img, params=None)
        except Exception as e:
            print ("Could not evaluate the given image.")
            print(e)

    end = time.time()
    print("-- ALL RESULTS --")
    print("Correct: ", correctly_recognized)
    print("Total: ", total_num_of_images)
    print(str(correctly_recognized) + " out of " + str(total_num_of_images) + " are recognized correctly.")
    print("-----------------")
    print("Correct (frontal): ", correctly_recognized_frontal)
    print("Total (frontal): ", total_num_of_frontal)
    print(str(correctly_recognized_frontal) + " out of " + str(total_num_of_frontal) + " are recognized correctly.")
    print("-----------------")

    print("Accuracy: ", correctly_recognized / total_num_of_images)
    print("Accuracy (frontal): ", correctly_recognized_frontal / total_num_of_frontal)
    print("It took: " + str(end - start) + " seconds")

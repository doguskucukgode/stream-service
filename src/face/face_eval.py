# External imports
import os
import sys
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

def read_image(path):
    try:
        img = cv2.imread(path)
        cv_encoded_img = cv2.imencode(".jpg", img)[1]
        encoded_img = base64.b64encode(cv_encoded_img)
        return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def evaluate_face(image_class, message):
    results = message['result']
    print("Results: ", results)
    if len(results) == 0:
        print("Result is empty")
    for res in results:
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
    ctx = zmq.Context(io_threads=1)
    socket = zmq_comm.init_client(ctx, address)

    eval_folder = "/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI_less/"
    image_list = os.listdir(eval_folder)
    image_list = [eval_folder + i for i in image_list]

    total_num_of_images = len(image_list)
    correctly_recognized = 0

    start = time.time()
    for im_path in image_list:
        filename = os.path.basename(im_path)
        print(filename)
        parts = filename.split('_')
        im_class = parts[0]
        im_index = parts[1]
        img = cv2.imread(im_path)
        encoded_img = read_image(im_path)
        socket.send(encoded_img)
        message = socket.recv_json()
        try:
            evaluation = evaluate_face(im_class, message)
            if evaluation is True:
                correctly_recognized += 1
            else:
                cv2.imwrite(
                    "/home/taylan/local/face-recognition/datasets/FEI_Brazil/wrong_classifications/wrong_"\
                     + str(im_class) + "-" + str(im_index), img, params=None\
                )
        except Exception as e:
            print ("Could not evaluate the given image.")
            print(e)

    end = time.time()
    print("-- ALL RESULTS --")
    print("Correct: ", correctly_recognized)
    print("Total: ", total_num_of_images)
    print(str(correctly_recognized) + " out of " + str(total_num_of_images) + " are recognized correctly.")
    print("-----------------")

    print("Accuracy: ", correctly_recognized / total_num_of_images)
    print("It took: " + str(end - start) + " seconds")

# External dependencies
import io
import zmq
import cv2
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import keras.backend.tensorflow_backend as K

# Internal dependencies
import zmq_comm
import car_conf


class Predict(object):
    name = ''
    score = -1

    def __init__(self, name, score):
        self.name = name
        self.score = score


def classifyIndices(data, preds, n):
    predValues = []
    for index, pred in enumerate(preds):
        predValues.append(Predict(data[str(index)], pred))

    predValuesSorted = sorted(predValues, key=lambda pred: pred.score, reverse=True)
    if(n <= len(predValuesSorted)):
        predValuesSorted = predValuesSorted[:n]
    return predValuesSorted


# You may edit the paths for the files in car_conf.py
def load_model_and_json():
    json_path = car_conf.classifier["model_folder"] + '/' + car_conf.classifier["classes_json"]
    print ("Loading JSON on path: ", json_path)
    json_file = open(json_path, 'r')
    json_read = json_file.read()
    loaded_model_json = json.loads(json_read)

    model_path = car_conf.classifier["model_folder"] + '/' + car_conf.classifier["model_file_name"]
    print ("Loading model on path: ", model_path)
    model = load_model(filepath=model_path)
    return model, loaded_model_json


def handle_requests(socket):
    # Set tensorflow configs
    tf_config = K.tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = car_conf.classifier["gpu_memory_frac"]
    K.set_session(K.tf.Session(config=tf_config))
    init = K.tf.global_variables_initializer()
    sess = K.get_session()
    sess.run(init)

    # Load model once
    model, loaded_model_json = load_model_and_json()
    print("Car classifier is ready to roll.")
    while True:
        result_dict = {}
        predict_list = []
        message = "OK"

        # Receive image and predict classes
        try:
            request = socket.recv()
            image = zmq_comm.decode_request(request)
            # Preprocess the image
            image = image * 1./255
            image = cv2.resize(image, (299, 299))
            image = image.reshape((1,) + image.shape)
            # Feed image to classifier
            preds = model.predict(image)[0]
            predict_list = classifyIndices(loaded_model_json, preds, car_conf.classifier["n"])
        except tf.errors.OpError as e:
            message = e.message
        except Exception as e:
            message = str(e)

        predictions = []
        tags = ["model", "score"]
        for index, pred in enumerate(predict_list):
            predictions.append(dict(zip(tags, [pred.name, str(pred.score)])))

        result_dict["result"] = predictions
        result_dict["message"] = message
        socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    try:
        host = car_conf.classifier["host"]
        port = car_conf.classifier["port"]
        tcp_address = zmq_comm.get_tcp_address(host, port)
        ctx = zmq.Context(io_threads=1)
        socket = zmq_comm.init_server(ctx, tcp_address)
        handle_requests(socket)
        print("Car classifier is being started on: ", tcp_address)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Socket closed properly.")
            socket.close()

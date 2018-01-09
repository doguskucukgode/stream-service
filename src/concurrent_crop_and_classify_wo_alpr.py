# External imports
import io
import os
import re
import cv2
import sys
import zmq
import uuid
import json
import copy
import base64
import operator
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import keras.backend.tensorflow_backend as K
from concurrent.futures import ProcessPoolExecutor, TimeoutError

# Internal imports
import zmq_comm
import car_conf
import plate_conf


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
print("Current dir: " + str(current_dir))
print("Parent dir: " + str(parent_dir))
print("Current working dir: " + str(os.getcwd()))

# The scripts that use darkflow must be in the darkflow directory
# The scripts that use SSD must be in the SSD directory
sys.path.insert(0, parent_dir + "/SSD-Tensorflow")
os.chdir(parent_dir + "/SSD-Tensorflow")

# For SSD detector
import cv2
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing
from nets import ssd_vgg_300, ssd_vgg_512, ssd_common, np_methods


# Since tensorflow does not allow for different memory usages for graphs used in the same process,
# we cannot make SSD use a different GPU fraction
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=car_conf.crcl["classifier_gpu_memory_frac"])
conf = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=conf)

slim = tf.contrib.slim
# Input placeholder.
net_shape = (0, 0)
net_to_use = car_conf.cropper['ssd-net']
ckpt_filename = car_conf.cropper['ssd-model-path']
net_shape = (300, 300) if net_to_use == 'ssd-300' else (512, 512)


class SSD_Bundle:
    def __init__(self, ssd_net, img_input, predictions, localisations, bbox_img, image_4d, ssd_anchors):
        self.ssd_net = ssd_net
        self.img_input = img_input
        self.predictions = predictions
        self.localisations = localisations
        self.bbox_img = bbox_img
        self.image_4d = image_4d
        self.ssd_anchors = ssd_anchors


def translate_ssd_label(x):
    return {
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
        6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
        12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }.get(x, 'other')    # 9 is default if x not found


def load_SSD_model():
    try:
        data_format = 'NHWC'
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        if net_to_use == 'ssd-300':
            ssd_net = ssd_vgg_300.SSDNet()
        else:
            ssd_net = ssd_vgg_512.SSDNet()

        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, ckpt_filename)

        # SSD default anchor boxes.
        ssd_anchors = ssd_net.anchors(net_shape)
        model = SSD_Bundle(ssd_net, img_input, predictions, localisations, bbox_img, image_4d, ssd_anchors)
        return model
    except Exception as e:
        message = "Could not load model."
        print(message + str(e))
        raise Exception(message)


# Main image processing routine of SSD.
def process_image(img, model, select_threshold=0.5, nms_threshold=.45, net_shape=net_shape):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run(\
        [model.image_4d, model.predictions, model.localisations, model.bbox_img],\
        feed_dict={model.img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, model.ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


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


# Load trained tensorflow model to classify cars
def load_model_and_json():
    json_path = car_conf.crcl["model_folder"] + '/' + car_conf.crcl["classes_json"]
    print ("Loading JSON on path: ", json_path)
    json_file = open(json_path, 'r')
    json_read = json_file.read()
    loaded_model_json = json.loads(json_read)

    model_path = car_conf.crcl["model_folder"] + '/' + car_conf.crcl["model_file_name"]
    print ("Loading model on path: ", model_path)
    model = load_model(filepath=model_path)
    return model, loaded_model_json


# Extracts objects from the given image and returns a list of predictions
def extract_objects(model, image):
    # Feed image to the network
    predictions = []
    try:
        rclasses, rscores, rbboxes = process_image(image, model)
        zipped = zip(rclasses, rscores, rbboxes)
        height = image.shape[0]
        width = image.shape[1]
        for (cl, score, box_coords) in zipped:
            obj_dict = {}
            # Translate integer label of SSDs to string
            obj_dict['label'] = translate_ssd_label(cl)
            # Skip this object if it's not one of these: 'car', 'bus', 'truck'
            if not obj_dict['label'] in ['car', 'bus']:
                continue
            # Convert float confidence values into string cos they are not JSON serializable
            obj_dict['confidence'] = str(score)
            ymin = int(box_coords[0] * height)
            xmin = int(box_coords[1] * width)
            ymax = int(box_coords[2] * height)
            xmax = int(box_coords[3] * width)
            obj_dict['topleft'] = {"x": xmin,"y": ymin}
            obj_dict['bottomright'] = {"x": xmax,"y": ymax}
            predictions.append(obj_dict)

        # print("Predictions: ", predictions)
        return predictions
    except AssertionError as e:
        message = "Could not convert given image into a numpy array."
    except AttributeError as e:
        message = "Make sure that given model is valid."
    except Exception as e:
        message = "Could not extract objects out of given image."
        print(message + str(e))
    raise Exception(message)


def crop_image(image, topleft, bottomright, confidence):
    x_margin_percentage = car_conf.crop_values['x_margin_percentage']
    y_margin_percentage = car_conf.crop_values['y_margin_percentage']

    y1 = topleft['y']
    y2 = bottomright['y']
    x1 = topleft['x']
    x2 = bottomright['x']
    width = x2 - x1
    height = y2 - y1
    min_length = height if height < width else width
    x_margin = int(min_length * x_margin_percentage)
    y_margin = int(min_length * y_margin_percentage)
    margined_X = 0 if (x1 - x_margin) < 0 else (x1 - x_margin)
    margined_Y = 0 if (y1 - y_margin) < 0 else (y1 - y_margin)
    margined_width = width + x_margin * 2
    margined_height = height + y_margin * 2

    actual_height, actual_width, channels = image.shape
    if confidence > car_conf.crop_values['min_confidence']:
        if margined_X + margined_width > actual_width:
            margined_width = actual_width - margined_X
        if margined_Y + margined_height > actual_height:
            margined_height = actual_height - margined_Y

        # Calculating bottomright coordinates
        margined_Y_BR = margined_Y + margined_height
        margined_X_BR = margined_X + margined_width
        # Returning cropped images with margined and original values
        return image[margined_Y:margined_Y_BR, margined_X:margined_X_BR], image[y1:y2, x1:x2]
    return None, None


#TODO: Add timeout logic here
def extract_plate(cropped, is_initialized):
    # Initialize zmq context and sockets when necessary
    if not is_initialized:
        print("Initializing plate sockets")
        extract_plate.ctx = zmq.Context(io_threads=1)
        plate_host = plate_conf.plate_server['host']
        plate_port = plate_conf.plate_server['port']

        plate_address = zmq_comm.get_tcp_address(plate_host, plate_port)
        extract_plate.plate_client = zmq_comm.init_client(extract_plate.ctx, plate_address)
        is_initialized = True

    found_plate = ""
    # Encode and send cropped car to detect plate location
    cv_encoded_img = cv2.imencode(".jpg", cropped)[1]
    encoded_img = base64.b64encode(cv_encoded_img)
    extract_plate.plate_client.send(encoded_img)
    plate_reply = extract_plate.plate_client.recv()
    plate_reply = json.loads(plate_reply.decode("utf-8"))
    # If there is an error, just return an empty plate
    if plate_reply['message'] != "OK":
        return found_plate, is_initialized

    found_plate = plate_reply['result']
    return found_plate, is_initialized


def handle_requests(ctx, socket):
    # Load SSD model
    ssd_model = load_SSD_model()

    # Load trained tensorflow car classifier and set tensorflow configs
    tf_config = K.tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = car_conf.crcl["classifier_gpu_memory_frac"]
    K.set_session(K.tf.Session(config=tf_config))

    init = K.tf.global_variables_initializer()
    sess = K.get_session()
    sess.run(init)

    # Define a concurrent future and an executor to be used in case plate recognition is enabled
    future1 = None
    executor = None
    is_initialized = False
    use_plate_recognition = car_conf.crcl["enable_plate_recognition"]
    if use_plate_recognition:
        # Initialize process executor once
        executor = ProcessPoolExecutor(max_workers=1)

    # Load model once
    car_classifier_model, car_classifier_loaded_model_json = load_model_and_json()
    print('Loaded both models successfully, ready to roll.')
    print('Server is started on:', tcp_address)

    while True:
        try:
            request = socket.recv()
            image = zmq_comm.decode_request(request)
            found_objects = []
            with isess.as_default():
                found_objects = extract_objects(ssd_model, image)

            clasifications = []
            with sess.as_default():
                for o in found_objects:
                    # Crop image according to empirical margin values
                    cropped, cropped_wo_margin = crop_image(image, o['topleft'], o['bottomright'], float(o['confidence']))
                    if cropped is None or cropped_wo_margin is None:
                        continue

                    found_plate = ""
                    predictions = []
                    filtered_candidates = []
                    # Run plate recognition in parallel while the main thread continues
                    # Note that, if you call 'future.result()' here, it just waits for process to end
                    # Also notice that, we are sending the original cropped image to plate extraction (a.k.a. no margins)
                    if use_plate_recognition:
                        future1 = executor.submit(extract_plate, cropped_wo_margin, is_initialized)

                    # Preprocess the image
                    cropped = cropped * 1./255
                    cropped = cv2.resize(cropped, (299, 299))
                    cropped = cropped.reshape((1,) + cropped.shape)
                    print("preprocessed")
                    # Feed image to classifier
                    preds = car_classifier_model.predict(cropped)[0]
                    predict_list = classifyIndices(
                        car_classifier_loaded_model_json,
                        preds,
                        car_conf.crcl["n"]
                    )
                    print("fed")
                    predictions = []
                    tags = ["model", "score"]
                    for index, p in enumerate(predict_list):
                        predictions.append(dict(zip(tags, [p.name, str(p.score)])))

                    print("predicted")
                    # Wait for plate recognition to finish its job
                    if use_plate_recognition and future1 is not None:
                        try:
                            found_plate, is_initialized = future1.result(timeout=car_conf.crcl["plate_service_timeout"])
                        except TimeoutError as e:
                            print("Could not get any respond from plate service. Timeout.")
                    print("waited for plate")
                    cl = {
                        'label' : o['label'],
                        'confidence' : o['confidence'],
                        'topleft' : o['topleft'],
                        'bottomright' : o['bottomright'],
                        'predictions' : predictions,
                        'plate' : found_plate
                    }
                    clasifications.append(cl)

            result_dict = {}
            result_dict["result"] = clasifications
            result_dict["message"] = "OK"
            # print(result_dict)
            socket.send_json(result_dict)
            print("sent")
        except Exception as e:
            result_dict = {}
            result_dict["result"] = []
            result_dict["message"] = str(e)
            socket.send_json(result_dict)
        finally:
            #TODO: Implement this
            pass


if __name__ == '__main__':
    socket = None
    try:
        host = car_conf.crcl["host"]
        port = car_conf.crcl["port"]
        tcp_address = zmq_comm.get_tcp_address(host, port)
        ctx = zmq.Context(io_threads=1)
        socket = zmq_comm.init_server(ctx, tcp_address)
        handle_requests(ctx, socket)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Closing the socket properly..")
            socket.close()

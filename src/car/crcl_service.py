# The latest version of car cropper and classification(CRCL) service
# Disregard other source files in this folder, they are basically the standalone versions
# You may use them if you need cropping or classifying only

# This version concurrently sends a request to the PlateService
# So be sure that PlateService is online, or it waits until the decided timeout
#for PlateService, hence slows down the whole car recognition service

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
from service import Service
import helper.zmq_comm as zmq_comm
# This is required because CRCL asks plate of the vehicles to plate service
from conf.plate_conf import PlateConfig
plate_configs = PlateConfig()

# The scripts that use SSD must be in the SSD directory
# Then import some stuff from SSD
module_dir = os.path.dirname(os.path.realpath(__file__))
source_dir = os.path.dirname(module_dir)
parent_dir = os.path.dirname(source_dir)
sys.path.insert(0, parent_dir + "/SSD-Tensorflow")
os.chdir(parent_dir + "/SSD-Tensorflow")
from preprocessing import ssd_vgg_preprocessing
from nets import ssd_vgg_300, ssd_vgg_512, ssd_common, np_methods

class SSD_Bundle:
    def __init__(self, ssd_net, img_input, predictions, localisations, bbox_img, image_4d, ssd_anchors):
        self.ssd_net = ssd_net
        self.img_input = img_input
        self.predictions = predictions
        self.localisations = localisations
        self.bbox_img = bbox_img
        self.image_4d = image_4d
        self.ssd_anchors = ssd_anchors

class Predict(object):
    name = ''
    score = -1

    def __init__(self, name, score):
        self.name = name
        self.score = score

class CRCLService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs.crcl["gpu_to_use"]
        # Since tensorflow does not allow for different memory usages for graphs used in the same process,
        # we cannot make SSD use a different GPU fraction
        self.gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.configs.crcl["classifier_gpu_memory_frac"]
        )
        self.tf_conf = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        self.isess = tf.Session(config=self.tf_conf)
        # Input placeholder.
        self.net_shape = (0, 0)
        self.net_to_use = self.configs.cropper['ssd-net']
        self.ckpt_filename = self.configs.cropper['ssd-model-path']
        self.net_shape = (300, 300) if self.net_to_use == 'ssd-300' else (512, 512)
        self.cropper_model = self.load_SSD_model()
        # Define a concurrent future and an executor to be used in case plate recognition is enabled
        self.use_plate_recognition = self.configs.crcl["enable_plate_recognition"]
        self.plate_future = None
        self.executor = ProcessPoolExecutor(max_workers=1) if self.use_plate_recognition else None
        self.handle_requests()

    def translate_label(self, x):
        return {
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
            6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
            12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }.get(x, 'other')    # 9 is default if x not found    pass

    def get_server_configs(self):
        return self.configs.crcl["host"], self.configs.crcl["port"]

    def load_SSD_model(self):
        try:
            data_format = 'NHWC'
            img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            # Evaluation pre-processing: resize to SSD net shape.
            image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                img_input, None, None, self.net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            image_4d = tf.expand_dims(image_pre, 0)

            # Define the SSD model.
            reuse = True if 'ssd_net' in locals() else None
            if self.net_to_use == 'ssd-300':
                ssd_net = ssd_vgg_300.SSDNet()
            else:
                ssd_net = ssd_vgg_512.SSDNet()

            with tf.contrib.slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
                predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

            # Restore SSD model.
            self.isess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.isess, self.ckpt_filename)

            # SSD default anchor boxes.
            ssd_anchors = ssd_net.anchors(self.net_shape)
            model = SSD_Bundle(ssd_net, img_input, predictions, localisations, bbox_img, image_4d, ssd_anchors)
            return model
        except Exception as e:
            message = "Could not load model."
            print(message + str(e))
            raise Exception(message)

    def process_image(self, img, select_threshold=0.5, nms_threshold=.45):
        """Main image processing routine."""
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([\
            self.cropper_model.image_4d, self.cropper_model.predictions,
            self.cropper_model.localisations, self.cropper_model.bbox_img],\
            feed_dict={self.cropper_model.img_input: img}
        )

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.cropper_model.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    def classifyIndices(self, data, preds, n):
        predValues = []
        for index, pred in enumerate(preds):
            predValues.append(Predict(data[str(index)], pred))

        predValuesSorted = sorted(predValues, key=lambda pred: pred.score, reverse=True)
        if(n <= len(predValuesSorted)):
            predValuesSorted = predValuesSorted[:n]
        return predValuesSorted

    def extract_objects(self, image):
        """Extracts objects from the given image and returns a list of predictions"""
        # Feed image to the network
        predictions = []
        try:
            rclasses, rscores, rbboxes = self.process_image(image)
            zipped = zip(rclasses, rscores, rbboxes)

            height = image.shape[0]
            width = image.shape[1]
            for (cl, score, box_coords) in zipped:
                obj_dict = {}
                # Translate integer label of SSDs to string
                obj_dict['label'] = self.translate_label(cl)
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

            return predictions
        except AssertionError as e:
            message = "Could not convert given image into a numpy array."
        except AttributeError as e:
            message = "Make sure that given model is valid."
        except Exception as e:
            message = "Could not extract objects out of given image."
            print(message + str(e))
        raise Exception(message)

    def load(self):
        json_path = self.configs.crcl["model_folder"] + '/' + self.configs.crcl["classes_json"]
        print ("Loading JSON on path: ", json_path)
        json_file = open(json_path, 'r')
        json_read = json_file.read()
        loaded_model_json = json.loads(json_read)

        model_path = self.configs.crcl["model_folder"] + '/' + self.configs.crcl["model_file_name"]
        print ("Loading model on path: ", model_path)
        model = load_model(filepath=model_path)
        return model, loaded_model_json

    def crop_image(self, image, topleft, bottomright, confidence):
        x_margin_percentage = self.configs.crop_values['x_margin_percentage']
        y_margin_percentage = self.configs.crop_values['y_margin_percentage']

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
        if confidence > self.configs.crop_values['min_confidence']:
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

    def handle_requests(self):
        # Load trained tensorflow car classifier and set tensorflow configs
        K.set_session(K.tf.Session(config=self.tf_conf))
        init = K.tf.global_variables_initializer()
        sess = K.get_session()
        sess.run(init)

        is_initialized = False
        self.classifier_model, self.classifier_json = self.load()
        print('Loaded both models successfully, ready to roll.')
        print('Server is started on:', self.address)

        while True:
            try:
                result_dict = {}
                classifications = []
                message = "OK"
                request = self.socket.recv()
                image = zmq_comm.decode_request(request)
                found_objects = []
                with self.isess.as_default():
                    found_objects = self.extract_objects(image)

                with sess.as_default():
                    for o in found_objects:
                        # Crop image according to empirical margin values
                        cropped, cropped_wo_margin = self.crop_image(
                            image, o['topleft'], o['bottomright'], float(o['confidence'])
                        )
                        if cropped is None or cropped_wo_margin is None:
                            continue

                        found_plate = ""
                        predictions = []
                        filtered_candidates = []
                        # Run plate recognition in parallel while the main thread continues
                        # Note that, if you call 'future.result()' here, it just waits for process to end
                        # Also notice that, we are sending the original cropped image to plate extraction (a.k.a. no margins)
                        if self.use_plate_recognition:
                            self.plate_future = self.executor.submit(extract_plate, cropped_wo_margin, is_initialized)

                        # Preprocess the image
                        cropped = cropped * 1./255
                        cropped = cv2.resize(cropped, (299, 299))
                        cropped = cropped.reshape((1,) + cropped.shape)
                        # Feed image to classifier
                        preds = self.classifier_model.predict(cropped)[0]
                        predict_list = self.classifyIndices(
                            self.classifier_json,
                            preds,
                            self.configs.crcl["n"]
                        )
                        predictions = []
                        tags = ["model", "score"]
                        for index, p in enumerate(predict_list):
                            predictions.append(dict(zip(tags, [p.name, str(p.score)])))

                        # Wait for plate recognition to finish its job
                        if self.use_plate_recognition and self.plate_future is not None:
                            try:
                                found_plate, is_initialized = self.plate_future.result(
                                    timeout=self.configs.crcl["plate_service_timeout"]
                                )
                            except TimeoutError as e:
                                print("Could not get any respond from plate service. Timeout.")
                        cl = {
                            'label' : o['label'],
                            'confidence' : o['confidence'],
                            'topleft' : o['topleft'],
                            'bottomright' : o['bottomright'],
                            'predictions' : predictions,
                            'plate' : found_plate
                        }
                        classifications.append(cl)
            except Exception as e:
                message = str(e)

            result_dict["result"] = classifications
            result_dict["message"] = message
            self.socket.send_json(result_dict)

    def terminate(self):
        print("Terminate called, executor is shutting down")
        if self.executor is not None:
            self.executor.shutdown(wait=False)

        if self.socket is not None:
            self.socket.close()
            print("Socket closed properly.")

# extract_plate() is defined here, because functions are only picklable
# if they are defined at the top-level of a module.
def extract_plate(cropped, is_initialized):
    # Initialize zmq context and sockets when necessary
    if not is_initialized:
        print("Initializing plate sockets")
        extract_plate.ctx = zmq.Context(io_threads=1)
        plate_host = plate_configs.plate_server["host"]
        plate_port = plate_configs.plate_server["port"]
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

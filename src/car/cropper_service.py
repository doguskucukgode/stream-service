# External dependencies
import io
import os
import cv2
import sys
import zmq
import json
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

# Internal imports
from service import Service
import helper.zmq_comm as zmq_comm

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

class CropperService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
        # TensorFlow session: grow memory when needed
        self.gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.configs.cropper["gpu_memory_frac"]
        )
        self.tf_conf = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        self.isess = tf.InteractiveSession(config=self.tf_conf)
        # Input placeholder.
        self.net_shape = (0, 0)
        self.net_to_use = self.configs.cropper['ssd-net']
        self.ckpt_filename = self.configs.cropper['ssd-model-path']
        self.net_shape = (300, 300) if self.net_to_use == 'ssd-300' else (512, 512)
        self.cropper_model = self.load()
        self.handle_requests()

    def translate_label(self, x):
        return {
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
            6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
            12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }.get(x, 'other')    # 9 is default if x not found    pass

    def get_server_configs(self):
        return self.configs.cropper["host"], self.configs.cropper["port"]

    def load(self):
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
            cropper_model = SSD_Bundle(ssd_net, img_input, predictions, localisations, bbox_img, image_4d, ssd_anchors)
            return cropper_model
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

    def handle_requests(self):
        print("Cropper service is started on: ", self.address)
        while True:
            result_dict = {}
            predictions = []
            message = "OK"
            try:
                request = self.socket.recv()
                image = zmq_comm.decode_request(request)
                predictions = self.extract_objects(image)
            except Exception as e:
                message = str(e)

            result_dict["result"] = predictions
            result_dict["message"] = message
            self.socket.send_json(result_dict)

import io
import os
import sys
import zmq
import json
import base64
import car_conf
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
print("Current dir: " + str(current_dir))
print("Parent dir: " + str(parent_dir))
print("Current working dir: " + str(os.getcwd()))

# The scripts that use SSD must be in the SSD directory
sys.path.insert(0, parent_dir + "/SSD-Tensorflow")
os.chdir(parent_dir + "/SSD-Tensorflow")

# For SSD detector
import cv2
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing
from nets import ssd_vgg_300, ssd_vgg_512, ssd_common, np_methods

# TensorFlow session: grow memory when needed
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=car_conf.cropper["gpu_memory_frac"])
conf = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=conf)

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


def translate_label(x):
    return {
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
        6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
        12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }.get(x, 'other')    # 9 is default if x not found


def load_model():
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


# Main image processing routine.
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
            obj_dict['label'] = translate_label(cl)
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


# Extracts the image from the received request
def decode_request(request):
    try:
        img = base64.b64decode(request)
        img = io.BytesIO(img)
        img = Image.open(img, mode="r")
        img.load()
        return np.asarray(img)
    except Exception as e:
        message = "Could not decode the received request."
        print(message + str(e))
        raise Exception(message)


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


def handle_requests(socket):
    model = load_model()
    print('The model is loaded without any errors.')
    while True:
        try:
            request = socket.recv()
            image = decode_request(request)
            predictions = extract_objects(model, image)
            # Build and send the json
            result_dict = {}
            result_dict["result"] = predictions
            result_dict["message"] = "OK"
            socket.send_json(result_dict)
        except Exception as e:
            result_dict = {}
            result_dict["result"] = []
            result_dict["message"] = str(e)
            socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    try:
        tcp_address = car_conf.get_tcp_address(car_conf.cropper["host"], car_conf.cropper["port"])
        socket = init_server(tcp_address)
        print('Server is started on:', tcp_address)
        handle_requests(socket)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Closing the socket properly..")
            socket.close()

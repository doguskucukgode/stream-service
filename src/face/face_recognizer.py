# External imports
import os
import sys
module_dir = os.path.dirname(os.path.realpath(__file__))
source_dir = os.path.dirname(module_dir)
sys.path.insert(0, source_dir)
import uuid

import cv2
import pickle
import numpy as np
import tensorflow as tf
from scipy import spatial

# Internal imports
import face.knn as faceknn
import face.preprocessing as fpreps
from conf.face_conf import FaceConfig
from face.face_detector import FaceDetector
from helper.time_stuff import measure_time

class FaceRecognizer():

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = FaceConfig.server["gpu_to_use"]
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.tf_conf = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        self.sess = tf.Session(config=self.tf_conf)
        self.images_placeholder = None
        self.embeddings = None
        self.embedding_size = None
        self.phase_train_placeholder = None
        self.model = None
        self.encoding_dict = None
        self.load()

    @measure_time
    def load(self):
        self.load_facenet_model()
        # If the encodings for faces are calculated before, load and use them
        if os.path.exists(FaceConfig.recognition["classifier_path"]):
            print("Loading all known faces from: ", FaceConfig.recognition["classifier_path"])
            self.encoding_dict = self.load_encodings(FaceConfig.recognition["classifier_path"])
        else:
            print("Could not load encodings from: ", FaceConfig.recognition["classifier_path"])
            print("Calculating encodings of known faces from scratch..")

            full_paths, base_paths = self.load_image_folder(
                FaceConfig.recognition['known_faces_folder']
            )
            self.encoding_dict = self.generate_encodings(full_paths, base_paths)
            print(self.encoding_dict)
            self.save_encodings(self.encoding_dict, FaceConfig.recognition["classifier_path"])

    def load_facenet_model(self):
        print("Loading facenet model..")
        self.sess.run(tf.global_variables_initializer())
        meta = tf.train.import_meta_graph(FaceConfig.recognition["facenet_meta"])
        meta.restore(self.sess, FaceConfig.recognition["facenet_ckpt"])
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print("Facenet model loaded.")

    def load_encodings(self, filepath):
        encodings = None
        with open(filepath, mode='rb') as f:
            encodings = pickle.load(f)
            print("Loaded the encodings from: ", filepath)
        return encodings

    def save_encodings(self, all_encodings, filepath):
        with open(filepath, mode='wb') as f:
            pickle.dump(all_encodings, f)
            print("Encodings are saved to the path: ", filepath)

    def generate_encodings(self, full_paths, base_paths):
         # Get input and output tensors
        emb_array = np.zeros((len(full_paths), self.embedding_size))

        detector = FaceDetector()
        # Run forward pass for each image to calculate embeddings
        for index, image_path in enumerate(full_paths):
            image = cv2.imread(image_path)
            faces = detector.detect(image)
            if len(faces) == 0:
                print("Could not detect a face in: ", image_path)
                continue
            face = faces[0]
            image = detector.align(image, face)
            image = fpreps.prewhiten(image)
            image = np.expand_dims(image, axis=0)
            feed_dict = { self.images_placeholder: image, self.phase_train_placeholder:False }
            emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
            emb_array[index] = emb

        return dict(zip(base_paths, emb_array))

    @measure_time
    def get_encoding(self, image):
        emb_array = np.zeros((1, self.embedding_size))
        image = fpreps.prewhiten(image)
        image = np.expand_dims(image, axis=0)
        feed_dict = { self.images_placeholder: image, self.phase_train_placeholder:False }
        encoding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return encoding

    def recognize(self, image):
        name = FaceConfig.recognition["not_recog_msg"]
        try:
            if FaceConfig.recognition["ml_algo"] == "knn":
                name = faceknn.recognize(self.encoding_dict, self.get_encoding(image))
            else:
                raise NotImplementedError("Provided ML algorithm is not implemented yet.")
        except Exception as e:
            print("Could not recognize the face due to: ", str(e))
        return name

    def load_image_folder(self, path_to_folder):
        full_paths = []
        base_paths = []
        files = os.listdir(path_to_folder)
        for img_file in files:
            full_paths.append(os.path.abspath(os.path.join(path_to_folder, img_file)))
            base_paths.append(os.path.basename(os.path.join(path_to_folder, img_file)))
        return full_paths, base_paths

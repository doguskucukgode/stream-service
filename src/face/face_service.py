# FaceService that detects and recognizes the given face

# External imports
from __future__ import division
import os
import cv2
import zmq
import json
import base64
import pickle
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import face_recognition
from collections import Counter

# Internal imports
from service import Service
import helper.zmq_comm as zmq_comm


class FaceService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs.server["gpu_to_use"]
        self.encodings, self.img_list = self.load()
        self.handle_requests()

    def get_server_configs(self):
        return self.configs.server["host"], self.configs.server["port"]

    def load(self):
        full_paths, base_paths = self.load_image_folder(
            self.configs.recognition['known_faces_folder']
        )
        encodings_file_path = self.configs.recognition['classifier_path']
        # If the encodings for faces are calculated before, load and use them
        if os.path.exists(encodings_file_path):
            print("Loading all known faces from: ", encodings_file_path)
            encodings = self.load_encodings(encodings_file_path)
        else:
            print("Could not load encodings from: ", encodings_file_path)
            print("Calculating encodings of known faces from scratch..")
            encodings = self.get_encodings_of_imgs(full_paths)
            self.save_encodings(encodings, encodings_file_path)
        return encodings, self.tag_images(base_paths)

    def save_encodings(self, all_encodings, filepath):
        with open(filepath, mode='wb') as f:
            pickle.dump(all_encodings, f)
            print("Encodings are saved to the path: ", filepath)

    def load_encodings(self, filepath):
        encodings = None
        with open(filepath, mode='rb') as f:
            encodings = pickle.load(f)
            print("Loaded the encodings from: ", filepath)
        return encodings

    def load_image_folder(self, path_to_folder):
        full_paths = []
        base_paths = []
        files = os.listdir(path_to_folder)
        for img_file in files:
            full_paths.append(os.path.abspath(os.path.join(path_to_folder, img_file)))
            base_paths.append(os.path.basename(os.path.join(path_to_folder, img_file)))
        return full_paths, base_paths

    def tag_images(self, img_list):
        for i in range(len(img_list)):
            img_list[i] = img_list[i].split("_")[0]
        return img_list

    def recognize_face(self, face_encoding, known_encodings, img_list, knn, similarity_threshold):
        counts = []
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        face_distances = np.asarray(face_distances)
        neighbour_n = face_distances.argsort()[:knn]
        for i in neighbour_n:
            counts.append(img_list[i])
        if face_distances[neighbour_n[knn-1]] > similarity_threshold:
            return face_conf.recognition["not_recog_msg"]
        counts = Counter(counts)
        result = counts[max(counts.keys(), key=(lambda k: counts[k]))]
        return counts.most_common()[0][0]

    def get_face_encoding(self, image):
        enc = face_recognition.face_encodings(image, num_jitters=10)
        return enc[0] if enc else []

    def get_encodings_of_imgs(self, image_path_list):
        all_encodings = []
        for img_path in image_path_list:
            print(img_path)
            image = face_recognition.load_image_file(img_path)
            encoding = self.get_face_encoding(image)
            all_encodings.append(encoding)
        return all_encodings

    def handle_requests(self):
        factor = self.configs.recognition["factor"]
        print("Face service is started on: ", self.address)
        while True:
            result_dict = {}
            face_locations = []
            face_encodings = []
            face_names = []
            message = "OK"

            final_results = []
            try:
                # Get image from socket
                request = self.socket.recv()
                image = zmq_comm.decode_request(request)
                # cv2.imwrite("./received.jpg", image)
                # Resize image to 1/factor size for faster face recognition processing
                small_img = cv2.resize(image, (0, 0), fx=1/factor, fy=1/factor)

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_img)
                face_encodings = face_recognition.face_encodings(small_img, face_locations)
                # print(face_encodings)
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    match_name = self.recognize_face(
                        face_encoding, self.encodings, self.img_list,
                        self.configs.recognition["knn"],
                        self.configs.recognition["similarity_threshold"]
                    )
                    face_names.append(match_name)

                predictions = []
                # Display the results
                for (top, right, bottom, left), match_name in zip(face_locations, face_names):
                    if match_name == self.configs.recognition["not_recog_msg"]:
                        continue
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top = int(top * factor)
                    bottom = int(bottom * factor)
                    left = int(left * factor)
                    right = int(right * factor)

                    face_dict = {}
                    face_dict['name'] = match_name
                    face_dict['topleft'] = {"x": left,"y": top}
                    face_dict['bottomright'] = {"x": right,"y": bottom}
                    # print(face_dict)
                    predictions.append(face_dict)

                final_results = predictions
            except Exception as e:
                message = str(e)

            result_dict["result"] = final_results
            result_dict["message"] = message
            self.socket.send_json(result_dict)

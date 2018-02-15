# External imports
import os
import cv2
import uuid

# Internal imports
from service import Service
import helper.zmq_comm as zmq_comm
from face.face_detector import FaceDetector
from face.face_recognizer import FaceRecognizer


class FaceService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs.server["gpu_to_use"]
        self.face_detector = None
        self.face_recognizer = None
        self.load()
        self.handle_requests()

    def get_server_configs(self):
        return self.configs.server["host"], self.configs.server["port"]

    def load(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

    def handle_requests(self):
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
                faces = self.face_detector.detect(image)
                face_labels = []
                for index, face in enumerate(faces):
                    aligned_n_cropped = self.face_detector.align(image, face)
                    face_id = self.face_recognizer.recognize(aligned_n_cropped)
                    face_labels.append(face_id)

                predictions = []
                # Display the results
                for dlib_rect, match_name in zip(faces, face_labels):
                    left, top, right, bottom = self.face_detector.rect_to_tuple(dlib_rect)
                    if match_name == self.configs.recognition["not_recog_msg"]:
                        continue
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                #     top = int(top * factor)
                #     bottom = int(bottom * factor)
                #     left = int(left * factor)
                #     right = int(right * factor)
                    face_dict = {}
                    face_dict['name'] = match_name
                    face_dict['topleft'] = {"x": left,"y": top}
                    face_dict['bottomright'] = {"x": right,"y": bottom}
                    predictions.append(face_dict)

                final_results = predictions
            except Exception as e:
                print("EXCEPTION: ", str(e))
                raise e

            result_dict["result"] = final_results
            result_dict["message"] = message
            self.socket.send_json(result_dict)

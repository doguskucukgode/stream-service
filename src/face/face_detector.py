# External imports
import os
import sys
module_dir = os.path.dirname(os.path.realpath(__file__))
source_dir = os.path.dirname(module_dir)
sys.path.insert(0, source_dir)

import cv2
import dlib
import numpy as np

# Internal imports
from conf.face_conf import FaceConfig
from helper.time_stuff import measure_time

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

class FaceDetector():

    def __init__(self):
        self.detector = None
        self.predictor = None
        self.load()

    @measure_time
    def load(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(FaceConfig.detection["predictor_path"])
        except Exception as e:
            print("Could not load face detector due to: ", str(e))
            raise e

    @measure_time
    def detect(self, img):
        faces = []
        try:
            # The 1 in the second argument indicates that we should upsample the image
            # 1 time.  This will make everything bigger and allow us to detect more
            # faces.
            faces = self.detector(img, 0)
        except Exception as e:
            print("Could not detect faces due to: ", str(e))
        return faces

    @measure_time
    def align(self, img, face):
        cropped = None
        height, width = img.shape[:2]
        try:
            shape = self.predictor(img, face)
            left_eye = self.extract_left_eye_center(shape)
            right_eye = self.extract_right_eye_center(shape)
            M = self.get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
            cropped = self.crop_image(rotated, face, FaceConfig.detection["margin"])
            desired_image_size = (FaceConfig.detection["resize_h"], FaceConfig.detection["resize_w"])
            cropped = cv2.resize(cropped, desired_image_size, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print("Could not align the given face due to: ", str(e))
        return cropped

    def rect_to_tuple(self, rect):
        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def extract_eye(self, shape, eye_indices):
        points = map(lambda i: shape.part(i), eye_indices)
        return list(points)

    def extract_eye_center(self, shape, eye_indices):
        points = self.extract_eye(shape, eye_indices)
        xs = map(lambda p: p.x, points)
        ys = map(lambda p: p.y, points)
        return sum(xs) // 6, sum(ys) // 6

    def extract_left_eye_center(self, shape):
        return self.extract_eye_center(shape, LEFT_EYE_INDICES)

    def extract_right_eye_center(self, shape):
        return self.extract_eye_center(shape, RIGHT_EYE_INDICES)

    def angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def get_rotation_matrix(self, p1, p2):
        angle = self.angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M

    def crop_image(self, image, det, margin=0):
        # Tries to add margin to the given images_placeholder
        # Returns the original rectangle whenever it fails
        left, top, right, bottom = self.rect_to_tuple(det)
        try:
            left = max(left - margin // 2, 0)
            top = max(top - margin // 2, 0)
            right = min(right + margin // 2, image.shape[1])
            bottom = min(bottom + margin // 2, image.shape[0])
            cropped = image[top:bottom, left:right, :]
        except:
            print("Could not add margin to the detected face. Returning original rectangle..")
            cropped = image[top:bottom, left:right, :]
        return cropped

if __name__ == '__main__':
    # Given an input and output dir, main func detects and aligns faces and saves them to output dir
    pic_folder = '/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI'
    output_dir = '/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI_aligned'
    detector = FaceDetector()

    files = os.listdir(pic_folder)
    files = [pic_folder + '/' + doc for doc in files]
    for f in files:
        filename = os.path.basename(f)
        img = cv2.imread(f)
        faces = detector.detect(img)
        for face in faces:
            cropped = detector.align(img, face)
            cv2.imwrite(output_dir + '/' + filename, cropped)

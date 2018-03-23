import os
import cv2
import time
import numpy as np


def load_classifier(classifier_path):
    plate_cascade = cv2.CascadeClassifier(classifier_path)
    return plate_cascade


def detect_plates(plate_classifier, img):
    plates = plate_classifier.detectMultiScale(img, 1.05, 6)
    print("Found plates: ", plates)
    return plates


if __name__ == '__main__':
    classifier_path = '/home/taylan/gitFolder/openalpr-2.3.0/runtime_data/region/eu.xml'
    img_folder_path = '/home/taylan/gitFolder/plate-deep-ocr/car_data/'
    img_name = 'web.jpg'
    margin = 0

    plate_classifier = load_classifier(classifier_path)

    imgs = os.listdir(img_folder_path)
    paths = [img_folder_path + i for i in imgs]

    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        start = time.time()
        plate_coords = detect_plates(plate_classifier, gray)
        end = time.time()
        print("Plate detection took: " + str((end - start) * 1000) + " ms")
        for (x, y, w, h) in plate_coords:
            cv2.rectangle(img, (x-margin, y-margin), (x+w+2*margin, y+h+2*margin), (255, 0, 0), 2)
        cv2.imwrite(p + '_detected.jpg', img)

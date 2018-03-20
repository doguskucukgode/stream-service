# This piece of code applies random forests technique to face recognition

# External imports
import os
import time
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Internal imports
from face_recognizer import FaceRecognizer

def load_encodings(filepath):
    encodings = None
    with open(filepath, mode='rb') as f:
        encodings = pickle.load(f)
        print("Loaded the encodings from: ", filepath)
    return encodings

def save_encodings(all_encodings, filepath):
    with open(filepath, mode='wb') as f:
        pickle.dump(all_encodings, f)
        print("Encodings are saved to the path: ", filepath)

if __name__ == '__main__':

    train_folder = "/home/taylan/gitFolder/stream-service/KnownPeeps"
    test_folder = "/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI_mini"

    train_encodings_path = "/home/taylan/gitFolder/stream-service/model/rf_train.pkl"
    test_encodings_path = "/home/taylan/gitFolder/stream-service/model/rf_test.pkl"

    train_encoding_dict = None
    test_encoding_dict = None

    if os.path.exists(train_encodings_path) and os.path.exists(test_encodings_path):
        train_encoding_dict = load_encodings(train_encodings_path)
        test_encoding_dict = load_encodings(test_encodings_path)
    else:
        face_recog = FaceRecognizer()
        print("Face recognizer initialized")
        start = time.time()
        train_full_paths, train_base_paths = face_recog.load_image_folder(train_folder)
        train_encoding_dict = face_recog.generate_encodings(train_full_paths, train_base_paths)
        save_encodings(train_encoding_dict, train_encodings_path)
        end = time.time()
        print("Generated encodings for train set. It took: ", end - start)
        print("Size of training images: ", len(train_full_paths))
        start = time.time()
        test_full_paths, test_base_paths = face_recog.load_image_folder(test_folder)
        test_encoding_dict = face_recog.generate_encodings(test_full_paths, test_base_paths)
        save_encodings(test_encoding_dict, test_encodings_path)
        end = time.time()
        print("Generated encodings for test set. It took: ", end - start)
        print("Size of test images: ", len(test_full_paths))
        print("Saved generated encodings")

    x_train, y_train = list(train_encoding_dict.values()), train_encoding_dict.keys()
    x_test, y_test = list(test_encoding_dict.values()), test_encoding_dict.keys()

    y_train_fixed = []
    y_test_fixed = []
    for k in y_train:
        y_train_fixed.append(k.split('_')[0])
    for k in y_test:
        y_test_fixed.append(k.split('_')[0])

    print("Fixed IDs")

    rf = RandomForestClassifier(
        n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto",
        max_leaf_nodes=None, bootstrap=True,
        oob_score=False, n_jobs=4, random_state=None, verbose=0,
        warm_start=False, class_weight=None
    )
    print("Training the classifier: ")
    start = time.time()
    rf.fit(x_train, y_train_fixed)
    end = time.time()
    print("Training done, it took: ", end - start)
    predicted = rf.predict(x_test)
    print("Predictions made")
    accuracy = accuracy_score(y_test_fixed, predicted)
    print("Accuracy: ", accuracy)

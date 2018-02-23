# External imports
import operator
import numpy as np
from collections import Counter

# Internal imports
from conf.face_conf import FaceConfig

def calculate_distances(encodings, unk_encoding):
    face_distances = {}
    for face_id, enc in encodings.items():
        diff = np.subtract(enc, unk_encoding)
        dist = np.sum(np.square(diff), 1)
        face_distances[face_id] = dist[0]
    return face_distances

def recognize(encodings, unk_encoding):
    # The values below determine how the algorithm behaves:
    # - k : number of nearest neighbours
    # - threshold: number of occurences of ids required to decide the class
    # - dist_threshold: distance value for the first guess
    # When k=3 and threshold=1, we check the 3NNs and require at least 2 of them to have the same id
    # Also, the distance of the first guess must be smaller than dist_threshold, there may be people
    #who are not in our dataset, who is similar to someone in the dataset
    k = FaceConfig.recognition["knn"]
    threshold = 1
    dist_threshold = 0.5

    face_distances = calculate_distances(encodings, unk_encoding)
    face_distances = sorted(face_distances.items(), key=operator.itemgetter(1))
    neighbour_ids = [face_id.split('_')[0] for face_id, enc in face_distances[0:k]]

    print("Closest class and distance: ", face_distances[0][0], face_distances[0][1])
    print("3-NN:", neighbour_ids)
    if face_distances[0][1] > dist_threshold:
        # print("Decided: -")
        return FaceConfig.recognition["not_recog_msg"]

    most_common_id, count = Counter(neighbour_ids).most_common(n=1)[0]
    name = FaceConfig.recognition["not_recog_msg"] if count <= threshold else most_common_id
    # print("Decided: ", name)
    return name

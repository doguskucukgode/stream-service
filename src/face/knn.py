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
    # When k=3 and threshold=1, we check the 3NNs and require at least 2 of them to have the same id
    k = FaceConfig.recognition["knn"]
    threshold = 1

    face_distances = calculate_distances(encodings, unk_encoding)
    face_distances = sorted(face_distances.items(), key=operator.itemgetter(1))
    neighbour_ids = [face_id.split('_')[0] for face_id, enc in face_distances[0:k]]
    most_common_id, count = Counter(neighbour_ids).most_common(n=1)[0]
    name = FaceConfig.recognition["not_recog_msg"] if count <= threshold else most_common_id
    return name

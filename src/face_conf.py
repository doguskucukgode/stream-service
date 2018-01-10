import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)
MODEL_FOLDER = BASE_FOLDER + "/model"

# Configs

server = {
    "host" : "127.0.0.1",
    "port" : "54444"
}

recognition = {
    "known_faces_folder" : "/home/taylan/local/face-recognition/datasets/FEI_Brazil/FEI_faces_known_not_cropped",
    "classifier_path" : MODEL_FOLDER + "/" + "encodings.pkl",
    "width" : 250,
    "height" : 250,
    "depth" : 3,
    "similarity_threshold" : 0.50,
    "factor" : 2,
    "knn" : 1,
    "not_recog_msg" : "-"
}

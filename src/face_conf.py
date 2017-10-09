import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)

# Configs

server = {
    "host" : "127.0.0.1",
    "port" : "54444"
}

recognition = {
    "known_faces_folder" : BASE_FOLDER + "/" + "KnownPeeps",
    "width" : 250,
    "height" : 250,
    "depth" : 3,
    "similarity_threshold" : 0.50,
    "encodings_file_name" : "encodings",
    "factor" : 2,
    "knn" : 1,
    "not_recog_msg" : "-"
}

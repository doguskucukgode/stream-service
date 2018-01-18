from base_config import BaseConfig

class FaceConfig(BaseConfig):

    server = {
        "host" : "127.0.0.1",
        "port" : "54444",
        "gpu_to_use" : "1"
    }

    recognition = {
        "known_faces_folder" : BaseConfig.base_folder + "/" + "KnownPeeps",
        "classifier_path" : BaseConfig.model_folder + "/" + "encodings.pkl",
        "width" : 250,
        "height" : 250,
        "depth" : 3,
        "similarity_threshold" : 0.50,
        "factor" : 2,
        "knn" : 1,
        "not_recog_msg" : "-"
    }

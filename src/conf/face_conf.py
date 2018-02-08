from base_config import BaseConfig

class FaceConfig(BaseConfig):

    server = {
        "host" : "127.0.0.1",
        "port" : "54444",
        "gpu_to_use" : "1"
    }

    detection = {
        "predictor_path" : BaseConfig.model_folder + "/" + "shape_predictor_68_face_landmarks.dat",
        "margin" : 32,
        "resize_h" : 160,
        "resize_w" : 160
    }

    recognition = {
        "known_faces_folder" : BaseConfig.base_folder + "/" + "KnownPeeps",
        "classifier_path" : BaseConfig.model_folder + "/" + "encodings.pkl",
        "facenet_ckpt" : BaseConfig.model_folder + "/" + "model-20170512-110547.ckpt-250000",
        "facenet_meta" : BaseConfig.model_folder + "/" + "model-20170512-110547.meta",
        "ml_algo" : "knn",
        "width" : 250,
        "height" : 250,
        "depth" : 3,
        "similarity_threshold" : 0.50,
        "factor" : 2,
        "knn" : 1,
        "not_recog_msg" : "-",
    }

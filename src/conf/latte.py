from conf.car_conf import CarConfig
from conf.face_conf import FaceConfig
from conf.plate_conf import PlateConfig
from conf.stream_conf import StreamConfig

class LattePlateConfig(PlateConfig):

    detector_server = {
        "gpu_to_use" : "0"
    }

    recognizer_server = {
        "gpu_to_use" : "0"
    }

    plate_server = {
        "gpu_to_use" : "0"
    }


class LatteStreamConfig(StreamConfig):

    service = {
        "gpu_to_use" : "0"
    }

    ipcam_demo = {
        "in_demo_mode" : False,
    }


class LatteCarConfig(CarConfig):

    cropper = {
        "gpu_memory_frac" : 0.2,
        "gpu_to_use" : "0"
    }

    classifier = {
        "gpu_memory_frac" : 0.2,
    }

    crcl = {
        "classifier_gpu_memory_frac" : 0.65,
        "enable_plate_recognition" : True,
        "plate_service_timeout" : 0.5,
        "gpu_to_use" : "0"
    }


class LatteFaceConfig(FaceConfig):

    server = {
        "gpu_to_use" : "0"
    }

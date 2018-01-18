from conf.car_conf import CarConfig
from conf.face_conf import FaceConfig
from conf.plate_conf import PlateConfig
from conf.stream_conf import StreamConfig

class LattePlateConfig(PlateConfig):
    PlateConfig.detector_server["gpu_to_use"] = "0"
    PlateConfig.recognizer_server["gpu_to_use"] = "0"
    PlateConfig.plate_server["gpu_to_use"] = "0"


class LatteStreamConfig(StreamConfig):
    StreamConfig.service["gpu_to_use"] = "0"
    StreamConfig.ipcam_demo["in_demo_mode"] = False


class LatteCarConfig(CarConfig):
    CarConfig.cropper["gpu_memory_frac"] = 0.2
    CarConfig.cropper["gpu_to_use"] = "0"
    CarConfig.classifier["gpu_memory_frac"] = 0.2
    CarConfig.crcl["classifier_gpu_memory_frac"] = 0.65
    CarConfig.crcl["enable_plate_recognition"] = True
    CarConfig.crcl["plate_service_timeout"] = 0.5
    CarConfig.crcl["gpu_to_use"] = "0"


class LatteFaceConfig(FaceConfig):
    FaceConfig.server["gpu_to_use"] = "0"

from base_config import BaseConfig

class PlateConfig(BaseConfig):

    detector_server = {
        "host" : "127.0.0.1",
        "port" : "42424",
        "gpu_to_use" : "1"
    }

    recognizer_server = {
        "host" : "127.0.0.1",
        "port" : "43434",
        "gpu_to_use" : "1"
    }

    plate_server = {
        "host" : "127.0.0.1",
        "port" : "41414",
        "gpu_to_use" : "1"
    }

    detection = {
        "classifier_path" : BaseConfig.model_folder + "/" + "plate_detector.xml",

        # The minimum detection strength determines how sure the detection algorithm must be before signaling that
        # a plate region exists.  Technically this corresponds to LBP nearest neighbors (e.g., how many detections
        # are clustered around the same area).  For example, 2 = very lenient, 9 = very strict.
        "detection_strictness" : 3,

        # detection_iteration_increase is the percentage that the LBP frame increases each iteration.
        # It must be greater than 1.0.  A value of 1.01 means increase by 1%, 1.10 increases it by 10% each time.
        # So a 1% increase would be ~10x slower than 10% to process, but it has a higher chance of landing
        # directly on the plate and getting a strong detection
        "detection_iteration_increase" : 1.1,

    }

    recognition = {
        # OpenALPR related configs
        "country" : "eu",
        "region" : "tr",
        "openalpr_conf_dir" : BaseConfig.source_folder + "/" + "openalpr.conf",
        "openalpr_runtime_data_dir" : BaseConfig.base_folder + "/openalpr-2.3.0/runtime_data",
        "top_n" : 30,

        "model_path" : BaseConfig.model_folder + "/" + "plate-recog-loss-rms-0.10.hdf5",

        # Default image width and height. Given images will be resized to these values and the fed to the network
        "image_width" : 128,
        "image_height" : 32,

        # Regex for invalid Turkish plates. If our regex does not match with a plate, then it is a good candidate.
        "invalid_tr_plate_regex" : '^(?![0-7][0-9]|80|81).*|(^[A-Z])|(^[0-9]{3})|(Q|X|W)|([A-Z]$)|([A-Z].*[0-9].*[A-Z])'
    }

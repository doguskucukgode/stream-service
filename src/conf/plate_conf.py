from base_config import BaseConfig

class PlateConfig(BaseConfig):

    detector_server = {
        "host" : "127.0.0.1",
        "port" : "42424",
        "gpu_to_use" : "0"
    }

    recognizer_server = {
        "host" : "127.0.0.1",
        "port" : "43434",
        "gpu_to_use" : "0"
    }

    plate_server = {
        "host" : "127.0.0.1",
        "port" : "41414",
        "gpu_to_use" : "0"
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

        # Regex for Turkish plates. If our regex matches with a plate, then it is a good candidate.
        "tr_plate_regex" : '^(?!00)^([0-7][0-9]|80|81)\s(([A-Z]{1}\s\d{4}$)|([A-Z]{2}\s\d{3,4}$)|([A-Z]{3}\s\d{2,3}$))'
    }

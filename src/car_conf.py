import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)


# Server configs
cropper = {
    "host" : "0.0.0.0",
    "port" : "55555",
    # SSD-detector options: ssd-300 or ssd-512
    # You should use appropriate net and pre-trained model obviously
    "ssd-net" : "ssd-300",
    "ssd-model-path": BASE_FOLDER + "/SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt",
    "gpu_memory_frac" : 0.2
}


crop_values = {
    "min_confidence" : 0.65,
    "x_margin_percentage" : 0.2,
    "y_margin_percentage" : 0.1
}


# Configs for car-model classifier
classifier = {
    "host" : "0.0.0.0",
    "port" : "55546",
    "source_folder" : SOURCE_FOLDER,
    "base_folder" : BASE_FOLDER,
    "model_folder" : BASE_FOLDER + '/' + 'model',
    "output_labels_file_name" : "output_labels.txt",
    "model_file_name" : "second.3.97-0.35.hdf5",
    "classes_json" : "classes.json",
    # First n predictions will be returned as result
    "n" : 5,
    "gpu_memory_frac" : 0.2,
    "min_confidence" : 0.75
}

training = {
    "train_data_dir" : "/home/dogus/Car_Recognition/keras/simple_train/data5/train",
    "test_data_dir" : "/home/dogus/Car_Recognition/keras/simple_train/data5/test",
    "num_of_classes" : 676,
    "batch_size" : 128,

    "aug_process_num" : 8,         # Number of processes to generate augmentations
    "skip_first_pass" : False,      # Skip first phase of fine tuning
    "use_custom_scheduler" : False, # Use the custom scheduler instead of ReduceLROnPlateau

    # To use a previous model, edit the following configs:
    "load_trained_model" : True,
    "path_trained_model" : "",

    # To load an InceptionV3 model with pre-trained weights
    "load_inception_model" : False,
    "path_trained_weights" : "",

    # Use the configs below whenever you'd like to change the last dense layer
    "change_classification_layer" : True,
    "neurons_in_new_fc_layer" : 679
}


# Configs for crop & classify, a seperate service
crcl = {
    "host" : "0.0.0.0",
    "port" : "54321",

    # Classifier config
    "source_folder" : SOURCE_FOLDER,
    "base_folder" : BASE_FOLDER,
    "model_folder" : BASE_FOLDER + '/' + 'model',
    "output_labels_file_name" : "output_labels.txt",
    "model_file_name" : "second.3.97-0.35.hdf5",
    "classes_json" : "classes.json",
    # First n predictions will be returned as result
    "n" : 5,

    # Since tensorflow does not allow for different memory usages for graphs used in the same process,
    # we cannot make SSD use a different GPU fraction and the config below does not work for crop and classify
    # It is used in cropper however
    # "ssd_gpu_memory_frac" : 0.3,
    "classifier_gpu_memory_frac" : 0.8,
    "enable_plate_recognition" : True,
    "plate_service_timeout" : 0.5
}

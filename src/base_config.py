import os

class BaseConfig:
    source_folder = os.path.dirname(os.path.realpath(__file__))
    base_folder = os.path.dirname(source_folder)
    model_folder = base_folder + "/model"

import sys
import inspect
import importlib
from base_config import BaseConfig

class DynamicImporter:

    def __init__(self):
        # WARNING: when the package name 'conf' is changed, the line below
        # must be changed as well.
        self.conf_module = importlib.import_module("conf")
        # print(self.conf_module.__dict__.keys())

    def import_dynamically(self, service_name, machine_id):
        if service_name in ["CRCLService", "CropperService"]:
            conf_file_name = "car_conf"
        elif service_name == "FaceService":
            conf_file_name = "face_conf"
        elif service_name == "PlateService":
            conf_file_name = "plate_conf"
        elif service_name == "StreamService":
            conf_file_name = "stream_conf"
        else:
            raise ValueError("Could not load configs for unknown service: ", service_name)

        # Getting the appropriate conf file from the conf package
        conf_file = getattr(self.conf_module, conf_file_name)
        # Retrieving the classes in the desired conf file
        classes_in_conf = inspect.getmembers(conf_file, predicate=inspect.isclass)
        # Finding a config class that is a subclass of BaseConfig
        klass = None
        for class_name, c in classes_in_conf:
            if issubclass(c, BaseConfig):
                klass = c

        if klass is None:
            raise ImportError("Could not find a suitable config file. Exiting..")
            sys.exit(status=1)

        if machine_id is not None:
            # If a machine config is required, look for it
            # Basically a similar importing process is repeated
            try:
                machine_conf_file = getattr(self.conf_module, machine_id)
                # print(machine_conf_file)
                classes_in_machine_conf = inspect.getmembers(machine_conf_file, predicate=inspect.isclass)
                # print(classes_in_machine_conf)
                for class_name, c in classes_in_machine_conf:
                    if issubclass(c, klass):
                        klass = c
            except Exception as e:
                print("Could not find configs for the given machine, due to: ", str(e))
                print("Using default configs..")

        instance = klass()
        return instance

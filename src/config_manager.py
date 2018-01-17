from conf import *
from helper.dynamic_importer import DynamicImporter

class ConfigManager:

    def __init__(self):
        self.loaded_configs = None
        self.dynamic_importer = DynamicImporter()

    def get_configurations(self, service, machine_id=None):
        """Read necessary configs if nothing is loaded. Return loaded configs."""
        if self.loaded_configs is None:
            try:
                print("Reading config files for: ", service, machine_id)
                self.loaded_configs = self.dynamic_importer.import_dynamically(
                    service, machine_id
                )
            except Exception as e:
                print("An error occured: ", str(e))
                raise ValueError(
                """Could not configure the given service instance. It is either
                a None instance or the given service instance type is unknown.
                Also check the given machine id."""
                )

        return self.loaded_configs

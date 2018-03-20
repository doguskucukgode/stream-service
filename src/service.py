# Service base class, all services (face recog., plate recog.) derives from it

# External imports
import zmq

# Internal imports
import helper.zmq_comm as zmqc
from config_manager import ConfigManager

class Service:

    def __init__(self, machine=None):
        try:
            self.config_man = ConfigManager()
            self.configure(machine)
            self.host, self.port = self.get_server_configs()
            self.address = zmqc.get_tcp_address(self.host, self.port)
            self.ctx = zmq.Context(io_threads=1)
            self.socket = zmqc.init_server(self.ctx, self.address)
        except Exception as e:
            print(str(e))
            self.terminate()

    def configure(self, machine=None):
        self.configs = self.config_man.get_configurations(self.__class__.__name__, machine)

    def get_server_configs(self):
        raise NotImplementedError

    def load(self, arg):
        raise NotImplementedError

    def handle_requests(self, arg):
        raise NotImplementedError

    def terminate(self):
        if self.socket is not None:
            self.socket.close()
            print("Socket closed properly.")

import os

# Helper functions
def get_tcp_address(host, port):
    return "tcp://" + host + ":" + port

SOURCE_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.dirname(SOURCE_FOLDER)

# Configs

server = {
    "host" : "127.0.0.1",
    "port" : "42424"
}

recognition = {
    "country" : "eu",
    "region" : "tr",
    "openalpr_conf_dir" : SOURCE_FOLDER + "/" + "openalpr.conf",
    "openalpr_runtime_data_dir" : BASE_FOLDER + "/openalpr-2.3.0/runtime_data",
    "top_n" : 30,
    "invalid_tr_plate_regex" : '^(?![0-7][0-9]|80|81).*|(^[A-Z])|(^[0-9]{3})|(Q|X|W)|([A-Z]$)|([A-Z].*[0-9].*[A-Z])'
}

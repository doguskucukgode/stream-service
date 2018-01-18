# External imports
import sys
import argparse

# Internal imports
from car.crcl_service import CRCLService
from car.cropper_service import CropperService
# from face.face_service import FaceService
from plate.plate_service import PlateService
from stream.stream_service import StreamService

if __name__ == '__main__':
    available_services = ["crcl", "cropper", "face", "plate", "stream"]

    parser = argparse.ArgumentParser(description="""
        Starts and manages a service according to the given service type.
    """)

    parser.add_argument("-t", default=None, type=str, dest="serv_type", \
        help="""
            Type of service you would like to start.
            Available options are:
        """ + " ".join(available_services)
    )

    parser.add_argument("-m", default=None, type=str, dest="machine_name", \
        help="""Name of the machine."""
    )

    parsed = parser.parse_args()
    if parsed.serv_type not in available_services:
        raise ValueError("Given service type is not available. Exiting..")
        sys.exit(status=1)

    try:
        s = None
        if parsed.serv_type == "crcl":
            print("Starting crcl service")
            s = CRCLService(machine=parsed.machine_name)
        elif parsed.serv_type == "cropper":
            print("Starting cropper service")
            s = CropperService(machine=parsed.machine_name)
        # elif parsed.serv_type == "face":
        #    print("Starting face service")
        #    s = FaceService(machine=parsed.machine_name)
        elif parsed.serv_type == "plate":
            print("Starting plate service")
            s = PlateService(machine=parsed.machine_name)
        elif parsed.serv_type == "stream":
            print("Starting stream service")
            s = StreamService(machine=parsed.machine_name)
        else:
            raise ValueError("An unexpected error has occured. Exiting..")
            sys.exit(status=1)
    except Exception as e:
        raise(e)
        sys.exit(status=1)
    finally:
        if s is not None:
            s.terminate()

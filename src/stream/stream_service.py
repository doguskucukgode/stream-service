# External imports
import os
import io
import zmq
import sys
import time
import cv2
import json
import random
import base64
import datetime
import requests
from requests.auth import HTTPDigestAuth
import xmltodict
import numpy as np
from PIL import Image
import multiprocessing
from subprocess import Popen, PIPE

# Internal imports
from service import Service
import helper.zmq_comm as zmq_comm
from conf.stream_conf import StreamConfig
from conf.car_conf import CarConfig

STREAM_SERVER_WOWZA = "wowza"
STREAM_SERVER_NGINX = "nginx"

class ReceivedInput:
    def __init__(self, input_type, read_url, read_stream, write_stream, action):
        self.input_type = input_type
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.action = action
        self.read_url = read_url
        self.write_url = StreamConfig.service["STREAM_URL"] + "/" + write_stream

class StreamProcess(multiprocessing.Process):
    def __init__(self, read_stream, write_stream, stype, sid, read_url, write_url):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.read_url = read_url
        self.write_url = write_url
        self.sid = sid
        self.stype = stype
        self.color_map = {}
        self.socket = None
        self.ctx = None
        self.read_popen = None
        self.write_popen = None
        self.width = StreamConfig.stream["width"]
        self.height = StreamConfig.stream["height"]
        self.channels = StreamConfig.stream["channels"]

    def run(self):
        self.ctx = zmq.Context(io_threads=1)
        if self.stype == StreamConfig.stream["TYPE_CAR_CLASSIFICATION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_CR_CL"])
        elif self.stype == StreamConfig.stream["TYPE_FACE_DETECTION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_FACE"])
        else:
            print("Invalid service type is given. Terminating the process..")
            self.shutdown()
        print("Client initialized")

        counter = 0
        tryCount = 0
        # Interval control decides whether we process the frame or sent a previously processed copy
        interval_ctl = 0
        decoded_msg = None
        print('Trying to read stream..')
        while not self.exit.is_set():
            try:
                # If subprocesses are not initialized, or they are dead, reinit them
                if self.read_popen is None or self.read_popen.poll() is not None:
                    self.read_popen = Popen([
                        '/home/dogus/ffmpeg_install/FFmpeg/ffmpeg', '-gpu', '0', '-hwaccel', 'cuvid',
                        '-i', self.read_url, '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-an', '-sn', '-f', 'image2pipe', '-'
                    ], stdout=PIPE, bufsize=10**8)
                    print("read_popen initialized.")

                if self.write_popen is None or self.write_popen.poll() is not None:
                    self.write_popen = Popen([
                         '/home/dogus/ffmpeg_install/FFmpeg/ffmpeg', '-gpu', '0', '-hwaccel', 'cuvid',
                        '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vf', 'scale=640:480',
                        '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url
                    ], stdin=PIPE)
                    print("write_popen initialized.")

                interval_ctl = (interval_ctl + 1) % 100
                # Read a single frame
                outs = self.read_popen.stdout.read(self.width * self.height * self.channels)
                im = np.fromstring(outs, dtype='uint8')
                im = im.reshape((self.height, self.width, self.channels))
                # The line below reverses the order of dimensions, what it does here: BGR -> RGB
                im = im[:,:,::-1]
                im = im.copy(order='C')
                # Process the received frame
                frame, decoded_msg, counter = self.process_frame(im, decoded_msg, interval_ctl, counter)
                # Reorder the image again for PIL Image format
                im = im[:,:,::-1]
                im = Image.fromarray(im)
                im.save(self.write_popen.stdin, 'JPEG')
            except ValueError as e:
                print("Received error: ", str(e))
                sys.stdout.flush()
                sys.stdin.flush()
                sys.stderr.flush()
                self.read_popen.kill()
                self.write_popen.kill()
                time.sleep(0.1)
                print("Retrying..")
                continue
            except Exception as e:
                raise e
        print("DONE")

    def process_frame(self, frame, decoded_msg, interval_ctl, counter):
        try:
            if interval_ctl % StreamConfig.stream["INTERVAL"] == 0:
                # Process the frame
                im = Image.fromarray(frame)
                encoded_img = self.read_image_base64_pil(im)
                self.socket.send(encoded_img)
                message = self.socket.recv()
                decoded_msg = json.loads(message.decode("utf-8"))
                results = decoded_msg['result']
                return_status = decoded_msg['message']

                if return_status == 'OK' and len(results) > 0:
                    # Annotate the frame and then send it
                    recognized_name = ''
                    if self.stype == StreamConfig.stream["TYPE_CAR_CLASSIFICATION"] :
                        annotated_img = self.annotate_crcl(frame, results)
                    elif self.stype == StreamConfig.stream["TYPE_FACE_DETECTION"] :
                        annotated_img, recognized_name = self.annotate_face(frame, results)
                    frame = annotated_img
                    counter = 0

                    # In demo mode, we save the recognized faces into a folder
                    if StreamConfig.ipcam_demo["in_demo_mode"]:
                        ts = time.time()
                        formatted_ts = datetime.datetime.fromtimestamp(ts).strftime(StreamConfig.ipcam_demo["timestamp_format"])
                        filename = formatted_ts + '_' + recognized_name + '.jpg'
                        path_to_save = StreamConfig.ipcam_demo['recog_save_path'] + "/" + filename
                        cv2.imwrite(path_to_save, annotated_img)
                else:
                    # If 'return_status' is not 'OK', send back the frame without any processing
                    pass
            else:
                # Send a copy
                if decoded_msg is not None and counter < StreamConfig.stream["COPY_COUNT"]:
                    #send with coordinate
                    counter += 1
                    results = decoded_msg['result']
                    return_status = decoded_msg['message']
                    if return_status == 'OK' and len(results) > 0:
                        # Annotate the frame and then send it
                        if self.stype == StreamConfig.stream["TYPE_CAR_CLASSIFICATION"] :
                            annotated_img = self.annotate_crcl(frame, results)
                        elif self.stype == StreamConfig.stream["TYPE_FACE_DETECTION"] :
                            annotated_img, _ = self.annotate_face(frame, results)
                        frame = annotated_img
                else:
                    decoded_msg = None
            return frame, decoded_msg, counter

        except Exception as e:
            print(str(e))

    def read_image_base64_pil(self, im):
        try:
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG")
            encoded_img = base64.b64encode(buffer.getvalue())
            return encoded_img
        except Exception as e:
            print ("Could not read the given image: " + str(e))

    def annotate_crcl(self, image, results):
        try:
            #results = json.loads(message.decode("utf-8"))['result']
            #print(results)
            for res in results:
                label = res['label']
                confidence = float(res['confidence'])
                predictions = res['predictions']
                if confidence > CarConfig.crop_values['min_confidence'] and\
                    float(predictions[0]['score']) > CarConfig.classifier['min_confidence']:
                    #color = (220, 152, 52)
                    if predictions[0]['model'] not in self.color_map:
                        self.color_map[predictions[0]['model']] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                    color = self.color_map[predictions[0]['model']]
                    topleft = res['topleft']
                    bottomright = res['bottomright']
                    topleft_x = topleft['x']
                    topleft_y = topleft['y']
                    bottomright_x = bottomright['x']
                    bottomright_y = bottomright['y']

                    cv2.rectangle(
                        image,
                        (topleft_x, topleft_y),
                        (bottomright_x, bottomright_y),
                        color,
                        4
                    )

                    plate = res['plate']
                    text = predictions[0]['model'] + ' - ' + str(predictions[0]['score'])
                    if plate != "":
                        cv2.rectangle(
                            image,
                            (topleft_x, topleft_y),
                            (topleft_x + 110, topleft_y + 20),
                            (255, 255, 255),
                            cv2.FILLED
                        )

                        cv2.putText(image, plate,
                            (topleft_x + 5, topleft_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )

                    text_x = topleft_x + 5
                    text_y = bottomright_y - 10
                    cv2.putText(image, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #(220, 220, 220),
                        color, 2, cv2.LINE_AA
                    )

        except Exception as e:
            print ("Could not annotate the given image.")
            print(str(e))
        return image

    def annotate_face(self, image, results):
        name = ''
        try:
            #results = json.loads(message.decode("utf-8"))['result']
            for res in results:
                topleft = res['topleft']
                bottomright = res['bottomright']
                name = res['name']

                cv2.rectangle(
                    image,
                    (int(topleft['x']), int(topleft['y'])),
                    (int(bottomright['x']), int(bottomright['y'])),
                    (255, 255, 255),
                    2
                )

                cv2.rectangle(
                    image,
                    (int(topleft['x']), int(bottomright['y']) - 25),
                    (int(bottomright['x']), int(bottomright['y'])),
                    (255, 255, 255),
                    cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image, name,
                    (int(topleft['x']) + 6, int(bottomright['y']) - 6),
                    font, 1.0, (0, 0, 0), 1
                )
        except Exception as e:
            print ("Could not annotate the given image.")
            print(str(e))
        return image, name

    def shutdown(self):
        print ("Shutdown initiated ", self.read_url)
        if self.socket is None:
            print("Socket is none!")
        else:
            self.socket.close()
            print("Socket closed properly")

        # Terminate subprocesses, if they are still running
        #TODO: Maybe flush stdin, stdout before terminating them?
        if self.read_popen is not None and self.read_popen.poll() is None:
            self.read_popen.terminate()
        if self.write_popen is not None and self.write_popen.poll() is None:
            self.write_popen.terminate()

        self.exit.set()


class StreamService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs.service["gpu_to_use"]
        self.stream_list = []
        self.load()
        self.handle_requests()

    def get_server_configs(self):
        return self.configs.service["host"], self.configs.service["port"]

    def load(self):
        #TODO: Populate this method with anything related to loading
        #StreamService does not seem to require such a method, but it's a pure
        #virtual method, that's why it needs to stay here
        pass

    def decode_request(self, request):
        try:
            json_data = json.loads(request.decode("utf-8"))
            return json_data
        except Exception as e:
            message = "Could not decode the received request."
            print(message + str(e))
            raise Exception(message)

    def decode_json(self, json_data):
        try:
            message = json_data["message"]
            received_input = ReceivedInput(
                int(message["type"]),
                str(message["read_url"]),
                str(message["read_stream"]),
                str(message["write_stream"]),
                int(message["action"])
            )
            return received_input
        except Exception as e:
            message = "Invalid JSON format."
            print(message + str(e))
            raise Exception(message)

    def remove_dead_processes(self):
        try:
            for stream in self.stream_list:
                if not stream.is_alive():
                    self.stream_list.remove(stream)
        except Exception as e:
            print("Could not remove dead process due to: ", str(e))

    def does_stream_exist(self, write_url):
        for stream in self.stream_list:
            if stream.sid == write_url:
                return True, stream
        return False, None

    def start_stream(self, received_input):
        message = "OK"
        result = None
        process = None
        print("START command " + received_input.read_url + " received")
        exists, stream = self.does_stream_exist(received_input.write_url)
        if exists:
            result = "Already in use"
            message = "FAIL"
            print(result)
            return message, result

        # If the process that we like to start is a new one
        if result is None:
            process = StreamProcess(
                received_input.read_stream,
                received_input.write_stream,
                received_input.input_type,
                received_input.write_url,
                received_input.read_url,
                received_input.write_url
            )
            process.start()
            self.stream_list.append(process)
            print(self.stream_list)
            result = process.write_stream
        return message, result

    def stop_stream(self, received_input):
        message = "OK"
        result = None
        process = None
        print("STOP command " + received_input.read_url + " received")
        exists, stream = self.does_stream_exist(received_input.write_url)
        if exists:
            print("Shutting down and removing process: ", stream)
            process = stream
            stream.shutdown()
            self.stream_list.remove(stream)
            return message, result

        if process is None:
            result = "Stream not found"
            message = "FAIL"
            print(result)
        else:
            result = process.write_stream
        return message, result

    def check_stream(self, received_input):
        message = "OK"
        result = None
        try:
            if self.configs.service["STREAM_SERVER"] == STREAM_SERVER_WOWZA:
                r = requests.get(
                    self.configs.wowza_stream_stat["url"],
                    headers=self.configs.wowza_stream_stat["headers"],
                    auth=HTTPDigestAuth(
                        self.configs.wowza_stream_stat["auth-user"],
                        self.configs.wowza_stream_stat["auth-pass"]
                    )
                )
                content = json.loads(r.content.decode("utf-8"))
                result = content["incomingStreams"]
            elif self.configs.service["STREAM_SERVER"] == STREAM_SERVER_NGINX:
                response = requests.get(self.configs.nginx_stream_stat["url"])
                o = xmltodict.parse(response.content)
                stats_json = o["rtmp"]["server"]["application"]["live"]
                values= []
                #check stream
                if 'stream' in stats_json:
                    stats_json = stats_json['stream']
                    if isinstance(stats_json,list):
                        for stat in stats_json:
                             values.append(stat)
                    else:
                        values.append(stats_json)
                result = values
        except Exception as e:
            print(e)
            result = "Request to check stream status failed."
            message = "FAIL"
            print(result)
        finally:
            return message, result

    def take_action(self, received_input):
        message = "OK"
        result = None
        try:
            if received_input.action == self.configs.actions["ACTION_START"]:
                message, result = self.start_stream(received_input)
            elif received_input.action == self.configs.actions["ACTION_STOP"]:
                message, result = self.stop_stream(received_input)
            elif received_input.action == self.configs.actions["ACTION_CHECK"]:
                message, result = self.check_stream(received_input)
            else:
                raise ValueError("Unexpected command received.")
            #TODO: None check for message and result
            return message, result
        except Exception as e:
            message = "Cannot decode input."
            print(message + str(e))
            raise Exception(message)

    def handle_requests(self):
        print("Stream server is started on: ", self.address)
        stream_list = []
        while True:
            result_dict = {}
            result = None
            try:
                request = self.socket.recv()
                json_data = self.decode_request(request)
                received_input = self.decode_json(json_data)
                self.remove_dead_processes()
                message, result = self.take_action(received_input)
            except Exception as e:
                print(e)
                result = str(e)
                message = "FAIL"
            finally:
                result_dict["result"] = result
                result_dict["message"] = message
                self.socket.send_json(result_dict)

    def terminate(self):
        print("Terminate called, all the child processes are shutting down..")
        for stream_process in self.stream_list:
            stream_process.shutdown()

        if self.socket is not None:
            self.socket.close()
            print("Socket closed properly.")

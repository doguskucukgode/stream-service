# External imports
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
        super().__init__(self)
        self.exit = multiprocessing.Event()
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.read_url = read_url
        self.write_url = write_url
        self.sid = sid
        self.stype = stype
        self.color_map = {}
        self.socket = None
        self.ctx = zmq.Context(io_threads=1)

    def run(self):
        if self.stype == StreamConfig.stream["TYPE_CAR_CLASSIFICATION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_CR_CL"])
        elif self.stype == StreamConfig.stream["TYPE_FACE_DETECTION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_FACE"])

        print("Client initialized")
        print("Connecting stream " + self.read_url + "...")
        cap = cv2.VideoCapture(self.read_url)
        print("Video Capture initialized " + self.read_url)
        #p =  Popen(['ffmpeg', '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url], stdin=PIPE)
        p =  Popen([
            StreamConfig.service['ffmpeg_path'], '-gpu', '1', '-hwaccel', 'cuvid',
            '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vf', 'scale=640:480',
            '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url
            ], stdin=PIPE
        )
        print("Popen initialized")

        tryCount = 0
        i = 0
        decoded_msg = None
        counter = 0
        while not self.exit.is_set():
            i = (i + 1) % 100
            ret, frame = cap.read()
            #print("Frame read, capture : " + str(ret))
            if ret is True:
                #print("Frame read ",i)
                try:
                    decoded_msg,counter = self.send_stream(frame, p, i, decoded_msg, counter)
                    tryCount = 0
                except Exception as e:
                    tryCount, cap, p, end_loop = self.reconnect(tryCount, p)
                    if(end_loop):
                        break
            else:
                tryCount, cap, p, end_loop = self.reconnect(tryCount, p)
                if(end_loop):
                    break
        p.stdin.close()
        p.wait()
        print (self.write_url + " exited!")

    def reconnect(self, tryCount, p):
        #wait for some time
        time.sleep(StreamConfig.stream["RECONNECT_TIME_OUT"])
        if tryCount == StreamConfig.stream["RECONNECT_TRY_COUNT"]:
            end_loop = True
            return tryCount, None, None, True
        else:
            p.stdin.close()
            p.wait()
            print("Reconnecting stream " + self.read_url + "...")
            cap = cv2.VideoCapture(self.read_url)
            print("Video Capture reinitialized " + self.read_url)
            #p =  Popen(['/home/dogus/ffmpeg_install/FFmpeg/ffmpeg','-gpu','0','-hwaccel','cuvid','-f', 'image2pipe','-vcodec', 'mjpeg','-i','-','-vcodec','h264','-an','-f','flv',self.write_url], stdin=PIPE)
            p =  Popen([
                'ffmpeg', '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-',
                '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url
                ], stdin=PIPE
            )
            print("Popen reinitialized")
            tryCount = tryCount + 1
            return tryCount, cap, p, False

    def send_stream(self, frame, popen, i, decoded_msg, counter):
        try:
            if i % stream_conf.stream["INTERVAL"] == 0:
                try:
                    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(cv2_im)
                    encoded_img = self.read_image_base64_pil(im)
                    self.socket.send(encoded_img)
                    message = self.socket.recv()
                    #print ("Received reply: ", message)
                except Exception as e:
                    print(str(e))

                decoded_msg = json.loads(message.decode("utf-8"))
                results = decoded_msg['result']
                return_status = decoded_msg['message']
                if return_status == 'OK' and len(results) > 0:
                    recognized_name = ''
                    if self.stype == stream_conf.stream["TYPE_CAR_CLASSIFICATION"] :
                        annotated_img = self.annotate_crcl(frame, results)
                    elif self.stype == stream_conf.stream["TYPE_FACE_DETECTION"] :
                        annotated_img, recognized_name = self.annotate_face(frame, results)
                    cv2_im = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(cv2_im)
                    counter = 0
                    '''
                    while counter < stream_conf.stream["COPY_COUNT"]:
                        im.save(popen.stdin, 'JPEG')
                        counter = counter + 1
                    '''

                    # In demo mode, we save the recognized faces in a
                    if stream_conf.ipcam_demo["in_demo_mode"]:
                        ts = time.time()
                        formatted_ts = datetime.datetime.fromtimestamp(ts)\
                            .strftime(stream_conf.ipcam_demo["timestamp_format"])
                        filename = formatted_ts + '_' + recognized_name + '.jpg'
                        path_to_save = stream_conf.ipcam_demo['recog_save_path']\
                            + "/" + filename
                        cv2.imwrite(path_to_save, annotated_img)

                    im.save(popen.stdin, 'JPEG')
                else:
                    im.save(popen.stdin, 'JPEG')
            else :
                if decoded_msg is not None and counter < stream_conf.stream["COPY_COUNT"]:
                    #send with coordinate
                    counter += 1
                    results = decoded_msg['result']
                    return_status = decoded_msg['message']
                    if return_status == 'OK' and len(results) > 0:
                        if self.stype == stream_conf.stream["TYPE_CAR_CLASSIFICATION"] :
                            annotated_img = self.annotate_crcl(frame, results)
                        elif self.stype == stream_conf.stream["TYPE_FACE_DETECTION"] :
                            annotated_img, recognized_name = self.annotate_face(frame, results)
                        cv2_im = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        im = Image.fromarray(cv2_im)
                        im.save(popen.stdin, 'JPEG')
                else:
                    decoded_msg = None
                    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(cv2_im)
                    im.save(popen.stdin, 'JPEG')
            return decoded_msg,counter
        except Exception as e:
            print(str(e))
            #Broken pipe error try reconnect
            raise e

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
                if confidence > car_conf.crop_values['min_confidence'] and\
                    float(predictions[0]['score']) > car_conf.classifier['min_confidence']:
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

    def annotate_face(self,image, results):
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
        self.exit.set()

class StreamService(Service):

    def __init__(self, machine=None):
        super().__init__(machine)
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

    def decode_request(request):
        try:
            json_data = json.loads(request.decode("utf-8"))
            return json_data
        except Exception as e:
            message = "Could not decode the received request."
            print(message + str(e))
            raise Exception(message)

    def decode_json(json_data):
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
            result = process.write_stream
        return message, result

    def stop_stream(self, received_input):
        message = "OK"
        result = None
        process = None
        print("STOP command " + received_input.read_url + " received")
        exists, stream = self.does_stream_exist(received_input.write_url)
        if exists:
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

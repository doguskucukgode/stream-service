# External configs
import io
import zmq
import sys
import time
import cv2
import json
import base64
import requests
import xmltodict
import requests
from requests.auth import HTTPDigestAuth
import numpy as np
from PIL import Image
import multiprocessing
from random import randint
from subprocess import Popen, PIPE
import random

# Internal configs
import service_config as serv_conf

STREAM_SERVER_WOWZA = "wowza"
STREAM_SERVER_NGINX = "nginx"

class ReceivedInput:
    def __init__(self, input_type, read_stream, write_stream, action):
        self.input_type = input_type
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.action = action
        self.read_url = serv_conf.service["STREAM_URL"] + "/" + read_stream
        self.write_url = serv_conf.service["STREAM_URL"] + "/" + write_stream


class StreamProcess(multiprocessing.Process):
    def __init__(self, read_stream, write_stream, stype, sid,read_url,write_url):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.read_url = read_url
        self.write_url = write_url
        self.sid = sid
        self.stype = stype
        self.color_map = {}


    def init_client(self,address):
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(address)
            return socket
        except Exception as e:
            print ("Could not initialize the client: " + str(e))


    def run(self):
        if self.stype == serv_conf.stream["TYPE_CAR_CLASSIFICATION"]:
            socket = self.init_client(serv_conf.service["ZMQ_URL_CR_CL"])
        elif self.stype == serv_conf.stream["TYPE_FACE_DETECTION"]:
            socket = self.init_client(serv_conf.service["ZMQ_URL_FACE"])

        print("Client initialized")
        print("Connecting stream " + self.read_url + "...")
        cap = cv2.VideoCapture(self.read_url)
        print("Video Capture initialized " + self.read_url)
        p =  Popen([serv_conf.service['ffmpeg_path'], '-gpu', '0', '-hwaccel', 'cuvid', '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url], stdin=PIPE)
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
                    decoded_msg,counter = self.send_stream(frame, p, i, socket, decoded_msg, counter)
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
        time.sleep(serv_conf.stream["RECONNECT_TIME_OUT"])
        if tryCount == serv_conf.stream["RECONNECT_TRY_COUNT"]:
            end_loop = True
            return tryCount, None, None, True
        else:
            p.stdin.close()
            p.wait()
            print("Reconnecting stream " + self.read_url + "...")
            cap = cv2.VideoCapture(self.read_url)
            print("Video Capture reinitialized " + self.read_url)
            #p =  Popen(['/home/dogus/ffmpeg_install/FFmpeg/ffmpeg','-gpu','0','-hwaccel','cuvid','-f', 'image2pipe','-vcodec', 'mjpeg','-i','-','-vcodec','h264','-an','-f','flv',self.write_url], stdin=PIPE)
            p =  Popen(['ffmpeg', '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url], stdin=PIPE)
            print("Popen reinitialized")
            tryCount = tryCount + 1
            return tryCount, cap, p, False


    def send_stream(self, frame, popen, i, socket, decoded_msg, counter):
        try:
            if i % serv_conf.stream["INTERVAL"] == 0:
                try:
                    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(cv2_im)
                    encoded_img = self.read_image_base64_pil(im)
                    socket.send(encoded_img)
                    message = socket.recv()
                    #print ("Received reply: ", message)
                except Exception as e:
                    print(str(e))

                decoded_msg = json.loads(message.decode("utf-8"))
                results = decoded_msg['result']
                return_status = decoded_msg['message']
                if return_status == 'OK' and len(results) > 0:
                    if self.stype == serv_conf.stream["TYPE_CAR_CLASSIFICATION"] :
                        annotated_img = self.annotate_crcl(frame, results)
                    elif self.stype == serv_conf.stream["TYPE_FACE_DETECTION"] :
                        annotated_img = self.annotate_face(frame, results)
                    cv2_im = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(cv2_im)
                    counter = 0
                    '''
                    while counter < serv_conf.stream["COPY_COUNT"]:
                        im.save(popen.stdin, 'JPEG')
                        counter = counter + 1
                    '''
                    im.save(popen.stdin, 'JPEG')
                else:
                    im.save(popen.stdin, 'JPEG')
            else :
                if decoded_msg is not None and counter < serv_conf.stream["COPY_COUNT"]:
                    #send with coordinate
                    counter += 1
                    results = decoded_msg['result']
                    return_status = decoded_msg['message']
                    if return_status == 'OK' and len(results) > 0:
                        if self.stype == serv_conf.stream["TYPE_CAR_CLASSIFICATION"] :
                            annotated_img = self.annotate_crcl(frame, results)
                        elif self.stype == serv_conf.stream["TYPE_FACE_DETECTION"] :
                            annotated_img = self.annotate_face(frame, results)
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
                if confidence > 0.5 and float(predictions[0]['score']) > 0.75:
                    #color = (220, 152, 52)
                    if predictions[0]['model'] not in self.color_map:
                        self.color_map[predictions[0]['model']] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                    color = self.color_map[predictions[0]['model']]
                    topleft = res['topleft']
                    bottomright = res['bottomright']

                    cv2.rectangle(
                        image,
                        (int(topleft['x']), int(topleft['y'])),
                        (int(bottomright['x']), int(bottomright['y'])),
                        color,
                        4
                    )

                    text = str(predictions[0]['model']) + ' - ' + str(predictions[0]['score'])
                    text_x = int(topleft['x']) + 5
                    text_y = int(bottomright['y']) - 10
                    cv2.putText(image, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #(220, 220, 220),
                        color, 2, cv2.LINE_AA
                    )

                    plate = res['plate']
                    if plate != "":
                        plate_x = int(topleft['x']) + 10
                        plate_y = int(topleft['y']) + 30
                        cv2.rectangle(
                            image,
                            (int(topleft['x']), int(topleft['y'])),
                            (int(topleft['x']) + 120, int(topleft['y'] + 50)),
                            (255, 255, 255),
                            cv2.FILLED
                        )

                        cv2.putText(image, plate, (plate_x, plate_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2, cv2.LINE_AA
                        )

        except Exception as e:
            print ("Could not annotate the given image.")
            print(str(e))
        return image


    def annotate_face(self,image, results):
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
        return image


    def shutdown(self):
        print ("Shutdown initiated ", self.read_url)
        self.exit.set()


def decode_request(request):
    try:
        #print(str(request.decode("utf-8")))
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
            str(message["read_stream"]),
            str(message["write_stream"]),
            int(message["action"])
        )
        return received_input
    except Exception as e:
        message = "Invalid format."
        print(message + str(e))
        raise Exception(message)


def decode_input(received_input,stream_list):
    try:
        #remove dead processes
        for stream in stream_list:
            if (not stream.is_alive()):
                stream_list.remove(stream)

        message = "OK"
        result = None
        #handle commands
        #Action start command
        if (received_input.action == serv_conf.actions["ACTION_START"]):
            process = None
            print("START command " + received_input.read_url + " received")
            for stream in stream_list:
                if (stream.sid == received_input.write_url):
                    result = "Already in use"
                    message = "FAIL"
                    print(result)
                    break

            if result is None:
                #start car classification process
                process = StreamProcess(
                    received_input.read_stream,
                    received_input.write_stream,
                    received_input.input_type,
                    received_input.write_url,
                    received_input.read_url,
                    received_input.write_url
                )
                process.start()
                stream_list.append(process)
                result = process.write_stream
        #Action stop command
        elif (received_input.action == serv_conf.actions["ACTION_STOP"]):
            process = None
            print("STOP command " + received_input.read_url + " received")
            for stream in stream_list:
                #print("Stream with sid : " + stream.sid)
                #print("received_input write_url : " + received_input.write_url)
                if (stream.sid == received_input.write_url):
                    process = stream
                    stream.shutdown()
                    stream_list.remove(stream)
                    break

            if process is None:
                result = "Stream not found"
                message = "FAIL"
                print(result)
            else:
                result = process.write_stream

        elif (received_input.action == serv_conf.actions["ACTION_CHECK"]):
            #print("CHECK command received")
            try:
                if(serv_conf.service["STREAM_SERVER"] == STREAM_SERVER_WOWZA):
                    r = requests.get(
                        serv_conf.wowza_stream_stat["url"],
                        headers=serv_conf.wowza_stream_stat["headers"],
                        auth=HTTPDigestAuth(
                            serv_conf.wowza_stream_stat["auth-user"],
                            serv_conf.wowza_stream_stat["auth-pass"]
                        )
                    )
                    content = json.loads(r.content.decode("utf-8"))
                    result = content["incomingStreams"]
                elif(serv_conf.service["STREAM_SERVER"] == STREAM_SERVER_NGINX):
                    response = requests.get(serv_conf.nginx_stream_stat["url"])
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

        return stream_list, message, result
    except Exception as e:
        message = "Cannot decode input."
        print(message + str(e))
        raise Exception(message)


def init_server(address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(address)
        return socket
    except Exception as e:
        message = "Could not initialize the server."
        print (message + str(e))
        raise Exception(message)


def handle_requests(socket):
    stream_list = []
    while True:
        try:
            request = socket.recv()
            json_data = decode_request(request)
            received_input = decode_json(json_data)
            #print("Before Size of stream_list : " , len(stream_list))
            stream_list, decode_message, result = decode_input(received_input, stream_list)
            #print("After Size of stream_list : " , len(stream_list))
            # Build and send the json
            result_dict = {}
            result_dict["result"] = result
            result_dict["message"] = decode_message
            socket.send_json(result_dict)
        except Exception as e:
            print(e)
            result_dict = {}
            result_dict["result"] = str(e)
            result_dict["message"] = "FAIL"
            socket.send_json(result_dict)


if __name__ == '__main__':
    socket = None
    try:
        tcp_address = serv_conf.get_tcp_address(serv_conf.service["host"], serv_conf.service["port"])
        socket = init_server(tcp_address)
        print('Server is started on:', tcp_address)
        handle_requests(socket)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Closing the socket properly..")
            socket.close()

import multiprocessing
import time
import sys
import service_config as config
import zmq
import cv2
import json
import numpy as np
import io
import base64
from subprocess import Popen, PIPE
from PIL import Image

TYPE_CAR_CLASSIFICATION = 0
TYPE_FACE_DETECTION = 1

ACTION_START = 0
ACTION_STOP = 1
INTERVAL = 20
COPY_COUNT = 8
RECONNECT_TIME_OUT = 10

class ReceivedInput:
    input_type = 0
    action = 0
    read_url = ""
    write_url = ""
    def __init__(self,input_type,read_url,write_url,action):
        self.input_type = input_type
        self.read_url = read_url
        self.write_url = write_url
        self.action = action


class StreamProcess(multiprocessing.Process):
    def __init__(self,read_url,write_url,type,id):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.read_url = read_url
        self.write_url = write_url
        self.id = id
        self.type = type


    def init_client(self,address):
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(address)
            return socket
        except Exception as e:
            print ("Could not initialize the client: " + str(e))


    def run(self):
        if self.type == TYPE_CAR_CLASSIFICATION:
            socket = self.init_client(config.service["ZMQ_URL_CR_CL"])
        elif self.type == TYPE_FACE_DETECTION:
            socket = self.init_client(config.service["ZMQ_URL_FACE"])
        print("Client initialized")
        print("Connecting stream "+self.read_url+"...")
        cap = cv2.VideoCapture(self.read_url)
        print("Video Capture initialized "+self.read_url)
        fps, duration = 24, 100
        p =  Popen(['ffmpeg', '-f', 'image2pipe','-vcodec', 'mjpeg','-i','-','-vcodec','h264','-an','-f','flv',self.write_url], stdin=PIPE)
        print("Popen initialized")
        tryCount = 0
        i = 0
        while not self.exit.is_set():
            i = (i + 1)%100
            ret, frame = cap.read()
            #print("Frame read, capture : " + str(ret))
            if ret is True:
                #print("Frame read ",i)
                tryCount = 0
                self.send_stream(frame, p,i,socket)
            else:
                #wait for some time
                time.sleep(RECONNECT_TIME_OUT)
                if tryCount == 10:
                    break
                print("Reconnecting stream "+self.read_url+"...")
                cap = cv2.VideoCapture(self.read_url)
                print("Video Capture initialized "+self.read_url)
                tryCount = tryCount +1
        p.stdin.close()
        p.wait()
        print ("You exited!")


    def send_stream(self,frame,popen,i,socket):
        if i%INTERVAL == 0:
            try:
                cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                im = Image.fromarray(cv2_im)
                encoded_img = self.read_image_base64_pil(im)
                socket.send(encoded_img)
                message = socket.recv()
                #print ("Received reply: ", message)
            except Exception as e:
                print(str(e))
            if self.type == TYPE_CAR_CLASSIFICATION :
                annotated_img = self.annotate_crcl(frame, message)
            elif self.type == TYPE_FACE_DETECTION :
                annotated_img = self.annotate_face(frame, message)
            cv2_im = cv2.cvtColor(annotated_img,cv2.COLOR_BGR2RGB)
            im = Image.fromarray(cv2_im)
            counter = 0
            while counter<COPY_COUNT:
                im.save(popen.stdin, 'JPEG')
                counter = counter + 1
        else :
            annotated_img = frame
            cv2_im = cv2.cvtColor(annotated_img,cv2.COLOR_BGR2RGB)
            im = Image.fromarray(cv2_im)
            im.save(popen.stdin, 'JPEG')


    def read_image_base64_pil(self,im):
        try:
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG")
            encoded_img = base64.b64encode(buffer.getvalue())
            return encoded_img
        except Exception as e:
            print ("Could not read the given image: " + str(e))

    def annotate_crcl(self,image, message):
        try:
            results = json.loads(message.decode("utf-8"))['result']
            #print(results)
            for res in results:
                label = res['label']
                confidence = float(res['confidence'])
                if confidence > 0.5:
                    if label == 'car' or label == 'bus' or label == 'truck':
                        topleft = res['topleft']
                        bottomright = res['bottomright']
                        cv2.rectangle(
                            image,
                            (int(topleft['x']), int(topleft['y'])),
                            (int(bottomright['x']), int(bottomright['y'])),
                            (0, 0, 0)
                        )
                        predictions = res['predictions']
                        text = str(predictions[0]['model']) + ' - ' + str(predictions[0]['score'])
                        text_x = int(topleft['x'])
                        text_y = int(topleft['y']) - 10
                        cv2.putText(
                            image,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA
                        )
        except Exception as e:
            print ("Could not annotate the given image.")
            print(str(e))
        return image


    def annotate_face(self,image, message):
        try:
            results = json.loads(message.decode("utf-8"))['result']
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
        json_data = json.loads(request)
        return json_data
    except Exception as e:
        message = "Could not decode the received request."
        print(message + str(e))
        raise Exception(message)

def decode_json(json_data):
    try:
        message = json_data["message"]
        received_input = ReceivedInput(int(message["type"]),
                                       str(message["url"]) + "/" + str(message["read_stream"]),
                                       str(message["url"]) + "/" + str(message["write_stream"]),
                                       int(message["action"]))
        return received_input
    except Exception as e:
        message = "Invalid format."
        print(message + str(e))
        raise Exception(message)

def decode_input(received_input,stream_list):
    try:
        if (received_input.action == ACTION_START):
            print("START command " + received_input.read_url + " received")
            message = None
            process = None
            for stream in stream_list:
                if (stream.id == received_input.write_url):
                    message = "Already in use"
                    print(message)
                    break
            if message is None:
                #start car classification process
                process = StreamProcess(received_input.read_url , received_input.write_url , received_input.input_type, received_input.write_url)
                process.start()
                stream_list.append(process)

        elif (received_input.action == ACTION_STOP):
            print("STOP command " + received_input.read_url + " received")
            process = None
            for stream in stream_list:
                #print("Stream with id : " + stream.id)
                if (stream.id == received_input.write_url):
                    process = stream
                    stream.shutdown()
                    stream_list.remove(stream)
                    break
            if process is None:
                message = "Stream not found"
                print(message)
        return process,stream_list,message
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
            request = socket.recv_json()
            json_data = decode_request(request)
            received_input = decode_json(json_data)
            print("Before Size of stream_list : " , len(stream_list))
            process,stream_list,decode_message = decode_input(received_input,stream_list)
            print("After Size of stream_list : " , len(stream_list))
            # Build and send the json
            result_dict = {}
            if process is not None:
                result_dict["result"] = process.id
            else:
                result_dict["result"] = decode_message
            result_dict["message"] = "OK"
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
        tcp_address = config.get_tcp_address(config.service["host"], config.service["port"])
        socket = init_server(tcp_address)
        print('Server is started on:', tcp_address)
        handle_requests(socket)
    except Exception as e:
        print(str(e))
    finally:
        if socket is not None:
            print("Closing the socket properly..")
            socket.close()

# A simple piece of code to try ffmpeg subprocesses

# External imports
import cv2
import zmq
import json
import time
import random
import base64
import datetime
import numpy as np
from PIL import Image
import multiprocessing
from subprocess import Popen, PIPE, check_output
import faulthandler
faulthandler.enable()

# Internal imports
import helper.zmq_comm as zmq_comm
from conf.car_conf import CarConfig
from conf.stream_conf import StreamConfig


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

    def run(self):
        self.ctx = zmq.Context(io_threads=1)
        if self.stype == StreamConfig.stream["TYPE_CAR_CLASSIFICATION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_CR_CL"])
        elif self.stype == StreamConfig.stream["TYPE_FACE_DETECTION"]:
            self.socket = zmq_comm.init_client(self.ctx, StreamConfig.service["ZMQ_URL_FACE"])
        else:
            print("Invalid service type is given. Terminating the process..")
            self.shutdown()
        time.sleep(0.1)
        print("Client initialized")

        self.read_popen = Popen([
            '/home/dogus/ffmpeg_install/FFmpeg/ffmpeg', '-gpu', '0', '-hwaccel', 'cuvid',
            '-i', self.read_url, '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-an', '-sn', '-f', 'image2pipe', '-'
        ], stdout=PIPE, bufsize=10**8)

        self.write_popen = Popen([
             '/home/dogus/ffmpeg_install/FFmpeg/ffmpeg', '-gpu', '0', '-hwaccel', 'cuvid',
            '-f', 'image2pipe','-vcodec', 'mjpeg', '-i', '-', '-vf', 'scale=640:480',
            '-vcodec', 'h264', '-an', '-f', 'flv', self.write_url
        ], stdin=PIPE)
        print("Subprocesses for reading and writing streams are initialized.")

        with self.read_popen as reading_proc, self.write_popen as writing_proc:
            counter = 0
            tryCount = 0
            # Interval control decides whether we process the frame or sent a previously processed copy
            interval_ctl = 0
            decoded_msg = None
            try:
                print('Trying to read stream..')
                while not self.exit.is_set():
                    interval_ctl = (interval_ctl + 1) % 100
                    # Read a single frame
                    outs = reading_proc.stdout.read(900*600*3)
                    im = np.fromstring(outs, dtype='uint8')
                    im = im.reshape((600, 900, 3))
                    # The line below reverses the order of dimensions, what it does here: BGR -> RGB
                    im = im[:,:,::-1]
                    # Process the received frame
                    frame, decoded_msg, counter = self.process_frame(im, decoded_msg, interval_ctl, counter)
                    # Reorder the image again for PIL Image format
                    im = im[:,:,::-1]
                    im = Image.fromarray(im)
                    im.save(writing_proc.stdin, 'JPEG')
                print('Stream became None.')
            except Exception as e:
                raise e

    def process_frame(self, frame, decoded_msg, interval_ctl, counter):
        try:
            if interval_ctl % StreamConfig.stream["INTERVAL"] == 0:
                # Process the frame
                encoded_img = base64.b64encode(frame)
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

    def terminate(self):
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

if __name__ == '__main__':
    read_url = 'rtmp://0.0.0.0:1935/live/tyStream'
    write_url = 'rtmp://0.0.0.0:1935/live/tyStream_CAR_RECOG'

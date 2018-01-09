# External imports
import os
import cv2
import zmq
import json
import base64
import face_conf

# Internal imports
import zmq_comm
import car_conf


def read_image_base64(path):
    try:
        with open(path, mode="rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())
            return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def read_image_new(path):
    try:
        img = cv2.imread(path)
        cv_encoded_img = cv2.imencode(".jpg", img)[1]
        encoded_img = base64.b64encode(cv_encoded_img)
        return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def annotate(image, message, out_path):
    results = message['result']
    for res in results:
        topleft = res['topleft']
        bottomright = res['bottomright']
        label = res['label']

        cv2.rectangle(
            image,
            (int(topleft['x']), int(topleft['y'])),
            (int(bottomright['x']), int(bottomright['y'])),
            (255, 255, 255)
        )

        cv2.putText(
            image,
            label,
            (int(topleft['x']), int(topleft['y'])),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 69, 0),
            6,
            cv2.LINE_AA
        )

    cv2.imwrite(out_path + '/result.jpg', image)
    print("Annotated image is written to: " + out_path + '/result.jpg')



def annotate_crcl(image, message, out_path):
    results = message['result']
    for res in results:
        label = res['label']
        confidence = float(res['confidence'])
        predictions = res['predictions']
        if confidence > car_conf.crop_values['min_confidence']:# and\
            # float(predictions[0]['score']) > car_conf.classifier['min_confidence']:
            print("Results fulfill the requirements, annotating the image..")
            topleft = res['topleft']
            bottomright = res['bottomright']
            topleft_x = int(topleft['x'])
            topleft_y = int(topleft['y'])
            bottomright_x = int(bottomright['x'])
            bottomright_y = int(bottomright['y'])
            image_width = bottomright_x - topleft_x
            image_height = bottomright_y - topleft_y

            cv2.rectangle(
                image,
                (topleft_x, topleft_y),
                (bottomright_x, bottomright_y),
                (255, 255, 255)
            )

            text = str(predictions[0]['model']) + ' - ' + str(predictions[0]['score'])
            text_x = int(topleft['x'])
            text_y = int(topleft['y']) - 10

            plate = res['plate']
            if plate != "":
                print("Detected plate: ", plate)
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

    cv2.imwrite(out_path + '/result.jpg', image)
    print("Annotated image is written to: " + out_path + '/result.jpg')


def annotate_face(image, message, out_path):
    results = message['result']
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

    cv2.imwrite(out_path + '/result.jpg', image)
    print("Annotated image is written to: " + out_path + '/result.jpg')


if __name__ == '__main__':
    # Set server info, you may use configs given in configurations
    host = "127.0.0.1"
    port = "41414"

    # Other stuff
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = "/home/taylan/Desktop/car_images/raw/plate-5.jpg"
    # input_path = "/home/taylan/gitFolder/face-recognition/KnownPeeps/taylan.jpg"

    image = cv2.imread(input_path, 1)
    # encoded_img = read_image_base64(input_path)
    encoded_img = read_image_new(input_path)
    tcp_address = zmq_comm.get_tcp_address(host, port)
    ctx = zmq.Context(io_threads=1)
    socket = zmq_comm.init_client(ctx, tcp_address)

    print ("Sending request..")
    socket.send(encoded_img)
    message = socket.recv_json()
    print ("Received reply: ", message)

    # WARNING: annotate works for cropper only, do not use it with classifier
    # use annotate() for cropper
    # use annotate_crcl() for cropper and classifier
    # use annotate_face() for face recog
    try:
        annotate_crcl(image, message, current_dir)
    except Exception as e:
        print ("Could not annotate the given image.")
        print(e)

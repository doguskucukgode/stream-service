import os
import cv2
import zmq
import json
import base64
import face_conf


def init_client(address):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(address)
        return socket
    except Exception as e:
        print ("Could not initialize the client: " + str(e))


def read_image_base64(path):
    try:
        with open(path, mode="rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())
            return encoded_img
    except Exception as e:
        print ("Could not read the given image: " + str(e))


def annotate(image, message, out_path):
    results = json.loads(message.decode("utf-8"))['result']
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
    results = json.loads(message.decode("utf-8"))['result']
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
                    (255, 255, 255)
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
                    1,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

    cv2.imwrite(out_path + '/result.jpg', image)
    print("Annotated image is written to: " + out_path + '/result.jpg')


def annotate_face(image, message, out_path):
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

    cv2.imwrite(out_path + '/result.jpg', image)
    print("Annotated image is written to: " + out_path + '/result.jpg')


if __name__ == '__main__':
    # Set server info, you may use configs given in configurations
    host = "127.0.0.1"
    port = "54321"

    # Other stuff
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = "/home/taylan/Desktop/lexus.jpg"
    # input_path = '/home/taylan/w_Python/oguzhan_face/Face/train/Alejandro_Toledo_0028.jpg'

    image = cv2.imread(input_path, 1)
    encoded_img = read_image_base64(input_path)
    address = face_conf.get_tcp_address(host, port)
    socket = init_client(address)

    print ("Sending request..")
    socket.send(encoded_img)
    message = socket.recv()
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

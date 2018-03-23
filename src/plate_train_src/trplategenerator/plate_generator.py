import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from . import augmentation

current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

city_codes = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
    '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
    '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
    '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
    '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
    '73', '74', '75', '76', '77', '78', '79', '80', '81'
]

alphabet = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'R', 'S', 'T', 'U', 'V', 'Y', 'Z'
]

possible_tr_plate_configs = [
    'PPX####',
    'PPXX###',
    'PPXX####',
    'PPXXX##',
    'PPXXX###'
]


plate_template_path = current_dir + '/plate_template.jpg'
output_path = current_dir + '/'

# Font related configs
font_file_path = current_dir + '/DIN-Bold.ttf'
font_size = 350
opacity = 255 # Between 0-255

# Original template image is 1640x359
topleft = (165, 30)
rightbottom = (1620, 330)


# Given an alphabet, generates a string of size n
def generate_n_char(alphabet, n):
    chars = ''
    for i in range(0, n):
        c = random.choice(alphabet)
        chars += c
    return chars


# Given a set of digits, generates a string of size n
def generate_n_numbers(digits, n):
    numbers = ''
    for i in range(0, n):
        d = random.choice(digits)
        numbers += d
    return numbers


def generate(font, plate_conf, city_code):
    num_of_chars = plate_conf.count('X')
    num_of_digits = plate_conf.count('#')

    chars = generate_n_char(alphabet, num_of_chars)
    numbers = generate_n_numbers(digits, num_of_digits)

    if plate_conf == 'PPX####':
        generated_plate_str = city_code + '   ' + chars + '   ' + numbers
    elif plate_conf == 'PPXX###':
        generated_plate_str = city_code + '   ' + chars + '   ' + numbers
    elif plate_conf == 'PPXX####':
        generated_plate_str = city_code + '  ' + chars + '  ' + numbers
    elif plate_conf == 'PPXXX##':
        generated_plate_str = city_code + '  ' + chars + '   ' + numbers
    elif plate_conf == 'PPXXX###':
        generated_plate_str = city_code + '  ' + chars + ' ' + numbers
    else:
        raise Exception('Given plate configuration is incorrect.')

    # print('Generated: ', ' '.join(generated_plate_str.split()))
    new_plate = Image.new(mode='RGBA', size=(1967, 300), color=(255, 255, 255))
    draw_plate = ImageDraw.Draw(new_plate, mode='RGBA')
    draw_plate.text((20, -70), generated_plate_str, fill=(0, 0, 0, opacity), font=font, anchor=None)
    new_plate = new_plate.resize((1967, 400))
    new_plate.thumbnail((1445, 300), Image.ANTIALIAS)
    im = Image.open(plate_template_path)
    im.paste(new_plate, topleft)
    return im, generated_plate_str


def plate_generator(should_augmentate):
    font = ImageFont.truetype(font=font_file_path, size=font_size, index=0, encoding="")
    # print("Using font: ", font_file_path)
    while True:
        plate_conf = random.choice(possible_tr_plate_configs)
        city_code = random.choice(city_codes)
        generated, generated_plate_str = generate(font, plate_conf, city_code)
        generated_plate_str = ' '.join(generated_plate_str.split())
        if should_augmentate:
            generated = np.array(generated)
            generated = augmentate_plates(generated)
            generated = Image.fromarray(generated.astype('uint8'), 'RGB')
        yield generated, generated_plate_str


# Given an image(np array), just applies augmentation on it
def augmentate_plates(img):
    pipe = augmentation.get_aug_pipeline()
    return pipe.augment_image(img)


def generate_random_image(w, h):
    im = np.random.rand(h, w, 3) * 255
    im = Image.fromarray(im.astype('uint8')).convert('RGB')
    return im


if __name__ == '__main__':
    n = 20
    pg = plate_generator(should_augmentate=True)
    for i in range(0, n):
        random_im = generate_random_image(1800, 500)
        generated, generated_plate_str = next(pg)
        generated_plate_str = ' '.join(generated_plate_str.split()).replace(" ", "_")
        # random_im.paste(generated, (80, 70))
        generated.save(output_path + generated_plate_str + '.jpg')

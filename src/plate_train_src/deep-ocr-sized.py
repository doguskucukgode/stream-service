import re
import os
from os.path import join
import sys
import cv2
import json
import time
import random
import datetime
import itertools
import editdistance
import numpy as np
import tensorflow as tf
import cairocffi as cairo

import pylab
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from trplategenerator import plate_generator

os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf_config = K.tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = K.tf.Session(config=tf_config)
K.set_session(sess)


alphabet = {
    0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
    10: 'A',  11: 'B',  12: 'C',  13: 'D',  14: 'E',  15: 'F',  16: 'G',  17: 'H',  18: 'I',
    19: 'J',  20: 'K',  21: 'L',  22: 'M',  23: 'N',  24: 'O',  25: 'P',  26: 'R',  27: 'S',
    28: 'T',  29: 'U',  30: 'V',  31: 'Y',  32: 'Z',  33: ' '
}

# Training variables
image_width = 128
image_height = 32
max_plate_len = 11
num_of_epochs = 250

model_folder = '/home/taylan/gitFolder/plate-deep-ocr/adam_model/'
real_data_path = '/home/taylan/gitFolder/plate-deep-ocr/data_duz/'
dataset_train_path = '/home/dogus/plate_recog/train'
dataset_test_path = '/home/dogus/plate_recog/test'

# If you'd like to load a pretrained model, edit 'load_trained_model' and 'model_name'
# Pretrained model that you wanna use must be put under 'model_folder'
load_trained_model = True
model_name = 'loss-rms-0.17.hdf5'
should_train = False
predict_unlabeled_data = False


def get_dataset(dir_path):
    dataset = []
    for entry in os.scandir(dir_path):
        if not entry.is_dir():
            label = entry.name.split('_')[0]
            dataset.append((entry.path,label))
    return dataset


dataset_train = get_dataset(dataset_train_path)
dataset_test = get_dataset(dataset_test_path)


def labels_to_text(text):
    return list(map(lambda x: alphabet[x], text))


def text_to_labels(labels):
    return list(map(lambda x: list(alphabet.values()).index(x), labels))


def is_valid_str(s):
    for ch in s:
        if not ch in alphabet.keys():
            return False
    return True


def pad_list(xlist, n):
    l = len(xlist)
    if l == n:
        return xlist
    elif l < n:
        i = n-l
        while i > 0:
            # Padding with space
            xlist.append(33)
            i -= 1
        return xlist
    else:
        raise Exception("cannot pad a list into a smaller one.")


class DataGenerator:
    def __init__(self, img_w, img_h, batch_size, downsample_factor, dataset, max_text_len=max_plate_len, n=10, apply_aug=False, shuffle=True):
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.apply_aug = apply_aug
        # Use a plate generator if you have no training data
        # self.plate_gen = plate_generator.plate_generator(True)
        # Use the given dataset if you have training data
        self.dataset = dataset
        # Amount of training/test data to generate at a time
        self.n = n
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.order_index = 0
        self.shuffle = shuffle


    # Generate large amount of data, to be served later via a generator
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        # Get 'self.n' many plates
        if self.shuffle:
            selected_img_tuples = self.get_random_tuples(self.n)
        else:
            selected_img_tuples = self.get_tuples_in_order(self.n)
        for i in range(self.n):
            img_path, label = selected_img_tuples[i]
            img = cv2.imread(img_path)
            if self.apply_aug:
                img = plate_generator.augmentate_plates(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(label)


    def get_output_size(self):
        return len(alphabet) + 1


    def get_tuples_in_order(self, n):
        dataset_size = len(self.dataset)
        tuples = []
        if n < dataset_size:
            last_index = min(self.order_index + n, dataset_size)
            ordered_indexes = list(range(self.order_index, last_index))
            print("Ordered index: ", ordered_indexes)
            for i in ordered_indexes:
                tuples.append(self.dataset[i])
            self.order_index = last_index
            return tuples
        else:
            raise Exception("Cannot supply " + str(n) + " training samples. Dataset is not that big!")


    def get_random_tuples(self, n):
        dataset_size = len(self.dataset)
        random_tuples = []
        if n < dataset_size:
            random_numbers = random.sample(range(0, dataset_size-1), n)
            for r in random_numbers:
                random_tuples.append(self.dataset[r])
            return random_tuples
        else:
            raise Exception("Cannot supply " + str(n) + " training samples. Dataset is not that big!")


    def next_sample(self):
        self.cur_index += 1
        # Re-generate 'self.n' many plates when all the plates are consumed
        if self.cur_index >= self.n:
            self.cur_index = 0
            self.build_data()
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]


    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                # print(pad_list(text_to_labels(text), max_plate_len))
                Y_data[i] = pad_list(text_to_labels(text), max_plate_len)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

# Checking whether we generate training samples correctly
datagen = DataGenerator(
    image_width, image_height, batch_size=5, downsample_factor=4,
    dataset=dataset_train, n=10, apply_aug=False, shuffle=True
)
datagen.build_data()

for inp, out in datagen.next_batch():
    print('Text generator output (data which will be fed into the neutral network):')
    print('1) the_input (image)')
    if K.image_data_format() == 'channels_first':
        img = inp['the_input'][0, 0, :, :]
    else:
        img = inp['the_input'][0, :, :, 0]

    plt.imshow(img.T, cmap='gray')
    plt.show()
    print('2) the_labels (plate number): %s is encoded as %s' %
          (labels_to_text(inp['the_labels'][0]), list(map(int, inp['the_labels'][0]))))
    print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
          (inp['input_length'][0], datagen.img_w))
    print('4) label_length (length of plate number): %d' % inp['label_length'][0])
    break


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, img_h, load=False):
    # Input Parameters
    input_shape = (img_w, img_h, 1)

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    batch_size = 32

    downsample_factor = pool_size ** 2
    train_datagen = DataGenerator(
        img_w, img_h, batch_size, downsample_factor,
        dataset=dataset_train, n=32, apply_aug=True, shuffle=True
    )
    train_datagen.build_data()

    val_datagen = DataGenerator(
        img_w, img_h, batch_size, downsample_factor,
        dataset=dataset_train, n=32, apply_aug=True, shuffle=True
    )
    val_datagen.build_data()

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation='relu', name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(train_datagen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[train_datagen.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up c`onvergence
    sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True, clipnorm=5)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6)
    rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=1e-6)

    if load:
        model_path = model_folder + model_name
        print("Loading model: ", model_path)
        model = load_model(model_path, compile=False)
    else:
        print("Not loading any models, training from scratch..")
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=rmsprop)

    if should_train:
        print("Not predicting real data.. training.")
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        checkpointer = ModelCheckpoint(
            filepath=model_folder + 'plate.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1,
            save_best_only=True,
            period=1
        )

        model.fit_generator(
            generator=train_datagen.next_batch(),
            steps_per_epoch=train_datagen.n,
            epochs=num_of_epochs,
            validation_data=val_datagen.next_batch(),
            validation_steps=val_datagen.n,
            verbose=1,
            callbacks=[checkpointer]
        )

    return model


model = train(image_width, image_height, load=load_trained_model)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))

        out_str_wo_gb = ''
        for c in out_best:
            if c < len(alphabet):
                out_str_wo_gb = out_str_wo_gb + alphabet[c] + '.'
        print(out_str_wo_gb)

        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(alphabet):
                outstr += alphabet[c]
        ret.append(outstr)
    return ret


def almost_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    else:
        count = 0
        for a, b in zip(w1, w2):
            if a != b :
                count += 1
            if count == 2:
                return False
        else:
            return True


if predict_unlabeled_data:
    print('Predicting real data..')
    real_data_files = os.listdir(real_data_path)
    real_data_paths = [real_data_path + p for p in real_data_files]
    images_to_feed = len(real_data_paths)

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    imgs = np.zeros((images_to_feed, image_height, image_width))
    X_data = np.ones([images_to_feed, image_width, image_height, 1])
    for index, f in enumerate(real_data_paths):
        print(f)
        test_im = cv2.imread(f)
        test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
        test_im = cv2.resize(test_im, (image_width, image_height))
        test_im = test_im.astype(np.float32)
        test_im /= 255
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        imgs[index, :, :] = test_im

    for j in range(images_to_feed):
        img = imgs[j]
        img = img.T
        if K.image_data_format() == 'channels_first':
            img = np.expand_dims(img, 0)
        else:
            img = np.expand_dims(img, -1)
        X_data[j] = img

    bs = X_data.shape[0]
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    print("Net out value: ", net_out_value)
    pred_texts = decode_batch(net_out_value)

    for i in range(bs):
        fig = plt.figure(figsize=(10, 10))
        outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, outer[1])
        fig.add_subplot(ax2)
        print('Predicted: %s\n' % (pred_texts[i]))
        img = X_data[i][:, :, 0].T
        ax1.set_title('Input img')
        ax1.imshow(img, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_title('Activations')
        ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
        ax2.set_yticks(list(range(len(alphabet) + 1)))
        ax2.set_yticklabels(alphabet)
        ax2.grid(False)
        for h in np.arange(-0.5, len(alphabet) + 1 + 0.5, 1):
            ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

        #ax.axvline(x, linestyle='--', color='k')
        plt.show()

else:
    test_datagen = DataGenerator(
        image_width, image_height, 500, 4,
        dataset=dataset_test, n=32, apply_aug=False, shuffle=False
    )
    test_datagen.build_data()

    start = time.time()
    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    correct_count = 0
    just_1_mistake_count = 0
    for inp_value, _ in test_datagen.next_batch():
        bs = inp_value['the_input'].shape[0]
        X_data = inp_value['the_input']
        net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
        print("Net out value: ", net_out_value)
        pred_texts = decode_batch(net_out_value)
        labels = inp_value['the_labels']
        texts = []
        print('LABELS: ', labels)
        for label in labels:
            text = ''.join(list(map(lambda x: alphabet[int(x)], label)))
            texts.append(text.strip())

        print(texts)

        for i in range(bs):
            if pred_texts[i] == texts[i]:
                print('Predicted: %s - True: %s' % (pred_texts[i], texts[i]))
                correct_count += 1
                just_1_mistake_count += 1
            elif almost_equal(pred_texts[i], texts[i]):
                just_1_mistake_count += 1
                print("INCORRECT! %s - True: %s" % (pred_texts[i], texts[i]))
            else:
                print("INCORRECT! %s - True: %s" % (pred_texts[i], texts[i]))


            # fig = plt.figure(figsize=(10, 10))
            # outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
            # ax1 = plt.Subplot(fig, outer[0])
            # fig.add_subplot(ax1)
            # ax2 = plt.Subplot(fig, outer[1])
            # fig.add_subplot(ax2)
            #
            # img = X_data[i][:, :, 0].T
            # ax1.set_title('Input img')
            # ax1.imshow(img, cmap='gray')
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax2.set_title('Activations')
            # ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
            # ax2.set_yticks(list(range(len(alphabet) + 1)))
            # ax2.set_yticklabels(alphabet)
            # ax2.grid(False)
            # for h in np.arange(-0.5, len(alphabet) + 1 + 0.5, 1):
            #     ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
            #
            # #ax.axvline(x, linestyle='--', color='k')
            # plt.show()

        print("Accuracy: ", (correct_count / bs))
        print("Accuracy with a single mistake: ", (just_1_mistake_count / bs))
        end = time.time()
        print("It took: " + str(end - start) + " sec")
        print("Average time per input: " + str((end - start) / bs))
        break

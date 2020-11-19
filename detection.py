import tensorflow as tf
import keras
from tensorflow import image
from keras.models import Sequential
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util.shape import view_as_windows
import IPython.display 
import os
import random
import cv2
import sys

import util

#--------------------------------------------- PARAMETERS ---------------------------------------------#

model_path = 'detection/model/'

neg_dir = 'dataset/non-faces/' #/content/drive/My Drive/SKRIPSI/Dataset/Random
pos_dir = 'dataset/faces/'

#training parameters
optimizer = 'SGD'

lr_12 = 1e-5
lr_24 = 1e-5
lr_48 = 1e-5

epoch_num = 100
val_freq = 1
val_split = 0.2
mini_batch_size = 32
#--------------------------------------------- UTILITY FUNCTIONS ---------------------------------------------#

def load_train(dim):
    # dim 24 -> 12, 24(resize from 12)
    hard_neg_dir = 'dataset/hard negative/'

    pos_file_list = [f for i,f in enumerate(os.listdir(pos_dir)) if f.endswith(".ppm")]

    if dim == 12:
        neg_file_list = [f for i, f in enumerate(os.listdir(neg_dir)) if f.endswith(".jpg")]
        print('Found '  + str(len(pos_file_list) + len(neg_file_list)) + ' images belonging to 2 clases.' )
    else: 
        hard_neg_dir = hard_neg_dir + str(dim) + '/'
        hard_neg_file_list_12 = [f for i,f in enumerate(os.listdir(hard_neg_dir)) if f.endswith(".jpg")]
        print('Found '  + str(len(pos_file_list) + len(hard_neg_file_list_12)) + ' images belonging to 2 clases.' )

    print('Loading training data...')

    pos_img = [Image.open(pos_dir + image) for image in pos_file_list]

    if dim == 12:
        neg_img = [Image.open(neg_dir + image) for image in neg_file_list]
    else:
        neg_img = [Image.open(hard_neg_dir + image) for image in hard_neg_file_list_12]

    pos_img_12 = [image.resize((12, 12), Image.BICUBIC) for image in pos_img]
    pos_img_12 = [img_to_array(image) for image in pos_img_12]
    pos_label = [ [0, 1] for i in range (0, len(pos_img_12))]

    neg_img_12 = [image.resize((12, 12), Image.BICUBIC) for image in neg_img]
    neg_img_12 = [img_to_array(image) for image in neg_img_12]
    neg_label = [ [1, 0] for i in range (0, len(neg_img_12))]

    train_img_12 = pos_img_12 + neg_img_12
    train_img_12 = np.array(train_img_12)
    train_label = pos_label + neg_label
    train_label = np.array(train_label)

    if dim == 12:
        return train_img_12, train_label
    
    if dim == 24:
        pos_img_24 = [image.resize((24, 24), Image.BICUBIC) for image in pos_img]
        pos_img_24 = [img_to_array(image) for image in pos_img_24]

        neg_img_24 = [image.resize((24, 24), Image.BICUBIC) for image in neg_img]
        neg_img_24 = [img_to_array(image) for image in neg_img_24]

        train_img_24 = pos_img_24 + neg_img_24
        train_img_24 = np.array(train_img_24)
    
        return train_img_12, train_label, train_img_24, train_label

    # dim 48 -> 12, 24, 48(resize from 24)
    if dim == 48:
        print('Loading training data...')

        pos_img_24 = [image.resize((24, 24), Image.BICUBIC) for image in pos_img]
        pos_img_24 = [img_to_array(image) for image in pos_img_24]
        pos_label_24 = [[0, 1] for i in range (0, len(pos_img_24))]

        # neg_img = [Image.open(hard_neg_dir + image) for image in hard_neg_file_list_24]
        neg_img_24 = [image.resize((24, 24), Image.BICUBIC) for image in neg_img]
        neg_img_24 = [img_to_array(image) for image in neg_img_24]
        neg_label_24 = [[1, 0] for i in range (0, len(neg_img_24))]

        train_img_24 = pos_img_24 + neg_img_24
        train_img_24 = np.array(train_img_24)
        train_label_24 = pos_label_24 + neg_label_24
        train_label_24 = np.array(train_label_24)

        # 48 data
        pos_img_48 = [image.resize((48, 48), Image.BICUBIC) for image in pos_img]
        pos_img_48 = [img_to_array(image) for image in pos_img_48]

        neg_img_48 = [image.resize((48, 48), Image.BICUBIC) for image in neg_img]
        neg_img_48 = [img_to_array(image) for image in neg_img_48]

        train_img_48 = pos_img_48 + neg_img_48
        train_img_48 = np.array(train_img_48)
        
        return train_img_12, train_label, train_img_24, train_label_24, train_img_48, train_label_24

#--------------------------------------------- MODEL CREATION ---------------------------------------------#

def create_net12():    
    input_12net = Input(shape=(12,12,3), name = 'input_12net')
    conv_12net = Conv2D(filters=16, kernel_size=3, strides=1, input_shape = (12,12,3), name = 'conv_12net')(input_12net)
    norm_1_12net = BatchNormalization(name = 'norm_1_12net')(conv_12net)
    act_1_12net = Activation('relu')(norm_1_12net)
    maxpool_12net = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_12net')(act_1_12net)
    act_1_12net = Activation('relu')(maxpool_12net)

    flatten_12net = Flatten(name = 'flatten_12net')(act_1_12net)
    fc_12net = Dense(units = 16, activation = 'relu', name = 'fc_12net')(flatten_12net)

    prediction_12net = Dense(units = 2, activation = 'softmax', name = 'prediction_12net')(fc_12net)

    net12 = keras.models.Model(inputs = input_12net, outputs = prediction_12net, name = 'net12')

    return net12

def create_net24():
    input_24net = Input(shape=(24,24,3), name= 'input_24net')
    conv_24net = Conv2D(filters=64, kernel_size=5, strides=1, input_shape = (24,24,3), name = 'conv_24net')(input_24net)
    maxpool_24net = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_24net')(conv_24net)
    act_1_24net = Activation('relu')(maxpool_24net)
    flatten_24net = Flatten(name = 'flatten_24net')(act_1_24net)
    fc_1_24net = Dense(units = 128, activation = 'relu', name = 'fc_1_24net')(flatten_24net)

    input_from_12net = Input(shape=(16,), name = 'input_from_12net')
    fc_concat_24 = concatenate([fc_1_24net, input_from_12net], axis = 1, name = 'fc_concat_24')
    fc_2_24net = Dense(units = 128+16, activation = 'relu', name = 'fc_2_24net')(fc_concat_24)
    prediction_24net = Dense(units = 2, activation = 'softmax', name = 'prediction_24net')(fc_2_24net)

    net24 = keras.models.Model(inputs = [input_24net, input_from_12net], outputs = prediction_24net, name ='net24')
    return net24

def create_net48():
    input_48net = Input(shape=(48,48,3), name= 'input_48net')
    conv_1_48net = Conv2D(filters=64, kernel_size=5, strides=1, input_shape = (48,48,3), name = 'conv_1_48net')(input_48net)
    maxpool_1_48net = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_1_24net')(conv_1_48net)
    act_1_48net = Activation('relu')(maxpool_1_48net)

    # normalization layer
    norm_1_48net = BatchNormalization(name = 'norm_1_48net')(act_1_48net)
    conv_2_48net = Conv2D(filters=64, kernel_size=5, strides=1, name = 'conv_2_48net')(norm_1_48net)

    # normalization layer
    norm_2_48net = BatchNormalization(name = 'norm_2_48net')(conv_2_48net)

    maxpool_2_48net = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_2_48net')(norm_2_48net)
    act_2_48net = Activation('relu')(maxpool_2_48net)
    flatten_48net = Flatten(name = 'flatten_48net')(act_2_48net)
    fc_1_48net = Dense(units = 256, activation = 'relu', name = 'fc_1_48net')(flatten_48net)

    input48_from_12net = Input(shape=(16,), name = 'input48_from_12net')
    input48_from_24net = Input(shape=(128,), name = 'input48_from_24net')

    fc_concat_48 = concatenate([fc_1_48net, input48_from_24net], axis = 1, name = 'fc_concat_48')
    fc_concat_2_48 = concatenate([fc_concat_48, input48_from_12net], axis = 1, name = 'fc_concat_2_48')

    fc_2_48net = Dense(units = (256+128+16), activation = 'relu', name = 'fc_2_48net')(fc_concat_2_48)
    prediction_48net = Dense(units = 2, activation = 'softmax', name = 'prediction_48net')(fc_2_48net)

    net48 = keras.models.Model(inputs = [input_48net, input48_from_24net, input48_from_12net], outputs = prediction_48net, name='net48')
    return net48

#--------------------------------------------- TRAINING ---------------------------------------------#

def train_12net():
    net12 = create_net12()
    net12.compile(
        optimizer = keras.optimizers.SGD(learning_rate = lr_12), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy', util.recall])

    train_img, train_label = load_train(12)
    train_img = np.array(train_img)
    train_label = np.array(train_label)

    print('Train start!')

    history = net12.fit(train_img, train_label,
              epochs = epoch_num,
              shuffle = True,
              batch_size = mini_batch_size,
              validation_split = val_split,
              validation_freq = val_freq)
    
    save_name = model_path + "12net.h5"

    net12.save(save_name)
    print("Saved model to disk")

    util.show_detect_train_history(history, '12net')

def train_24net():
    net24 = create_net24()
    net24.compile(
    optimizer = keras.optimizers.SGD(learning_rate = lr_24), 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy', util.recall])

    
    train_img12, train_label12, train_img24, train_label24 = load_train(24)
    train_img12 = np.array(train_img12)
    train_label12 = np.array(train_label12)
    train_img24 = np.array(train_img24)
    train_label24 = np.array(train_label24)
    
    print('Predicting 12net fc...')

    name12 = model_path + '12net.h5'
    from_12 = load_model(name12, custom_objects={'recall': util.recall})

    # output from 12 net training
    out_fc_12net_model = Model(
        inputs = from_12.input,
        outputs = from_12.get_layer(name = 'fc_12net').output)
    out_fc_12net = out_fc_12net_model.predict(train_img12)

    print('Train start!')

    history = net24.fit({'input_24net': train_img24, 'input_from_12net': out_fc_12net}, 
              {'prediction_24net' : train_label24}, 
              epochs = epoch_num, 
              batch_size = mini_batch_size,
              validation_split = val_split,
              shuffle = True,
              validation_freq = val_freq)

    save_name = model_path + "24net.h5" 
    
    net24.save(save_name)
    print("Saved model to disk")

    util.show_detect_train_history(history, '24net')

def train_48net():
    net48 = create_net48()
    net48.compile(
    optimizer = keras.optimizers.SGD(learning_rate = lr_48), 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy', util.recall])

    train_img12, train_label12, train_img24, train_label24, train_img48, train_label48 = load_train(48)
    train_img12 = np.array(train_img12)
    train_img24 = np.array(train_img24)
    train_img48 = np.array(train_img48)

    train_label12 = np.array(train_label12)
    train_label24 = np.array(train_label24)
    train_label48 = np.array(train_label48)

    print('Train start!')

    name12 = model_path + '12net.h5'
    name24 = model_path + '24net.h5'

    from_12 = load_model(name12, custom_objects={'recall': util.recall})
    from_24 = load_model(name24, custom_objects={'recall': util.recall})

    # output from 12 net training
    out_fc_12net_model = Model(
        inputs = from_12.input,
        outputs = from_12.get_layer(name = 'fc_12net').output)
    out_fc_12net = out_fc_12net_model.predict(train_img12)

    # output from 24 net training
    out_fc_24net_model = Model(
        inputs = from_24.input,
        outputs = from_24.get_layer(name = 'fc_1_24net').output)
    out_fc_24net = out_fc_24net_model.predict(
        {'input_24net': train_img24, 
         'input_from_12net': out_fc_12net})

    history = net48.fit({'input_48net': train_img48, 
               'input48_from_24net': out_fc_24net, 
               'input48_from_12net': out_fc_12net}, 
              {'prediction_48net' : train_label48}, 
              epochs = epoch_num, 
              batch_size = mini_batch_size,
              validation_split = val_split,
              shuffle = True,
              validation_freq = val_freq)
    
    save_name = model_path + "48net.h5"

    net48.save(save_name)
    print("Saved model to disk")

    util.show_detect_train_history(history, '48net')

if __name__ == '__main__':
    globals()[sys.argv[1]]()
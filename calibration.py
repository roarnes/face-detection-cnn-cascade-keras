import tensorflow
import keras
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
import cv2
from PIL import Image
import os
import sys

import util

#--------------------------------------------- PARAMETERS ---------------------------------------------#

epoch_num = 50
learning_rate = 1e-3
val_split = 0.3
mini_batch_size = 8

calib_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
calib_off_x = [-0.17, 0., 0.17]
calib_off_y = [-0.17, 0., 0.17]
calib_pattern_num = len(calib_scale) * len(calib_off_x) * len(calib_off_y)

model_path = 'calibration/model/'

#--------------------------------------------- LOAD DATASET ---------------------------------------------#

def load_calib_train(dim):
    folder = 'dataset/calib/0/'
    data_path = 'dataset/calib/'
    
    file_list = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    file_list = file_list[:100]
    print(len(file_list))

    train_img = list()
    train_label = list()

    for i, image in enumerate(file_list):
        if i%1000 == 0:
            print('Processed ', str(i), ' images')
        for j in range(0, 45):
            label = [0 for i in range (0, 45)]
            fp = os.path.join(data_path, str(j) + '/')
            
            img = Image.open(fp + image)
            img = img.resize((int(dim), int(dim)), Image.BICUBIC)
            img = img_to_array(img)
            train_img.append(img)
            
            label[j] = 1
            train_label.append(label)

    print(len(train_img), len(train_label))
    train_img = np.array(train_img)
    train_label = np.array(train_label)
    return train_img, train_label

#--------------------------------------------- MODEL CREATION ---------------------------------------------#


def create_calib12():
    input_12calib = Input(shape=(12,12,3), name = 'input_12calib')
    conv_12calib = Conv2D(filters=16, kernel_size=3, strides=1, input_shape = (12,12,3), activation = 'relu', name = 'conv_12calib')(input_12calib)
    maxpool_12calib = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_12calib')(conv_12calib)
    flatten_12calib = Flatten(name = 'flatten_12calib')(maxpool_12calib)
    fc_12calib = Dense(units = 128, activation = 'relu', name = 'fc_12calib')(flatten_12calib)
    prediction_12calib = Dense(units = 45, activation = 'softmax', name = 'prediction_12calib')(fc_12calib)

    calib12 = keras.models.Model(inputs = input_12calib, outputs = prediction_12calib, name = 'calib12')

    return calib12

def create_calib24():
    input_24calib = Input(shape=(24,24,3), name = 'input_24calib')
    conv_24calib = Conv2D(filters=32, kernel_size=5, strides=1, input_shape = (24,24,3), activation = 'relu', name = 'conv_24calib')(input_24calib)
    maxpool_24calib = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_24calib')(conv_24calib)
    flatten_24calib = Flatten(name = 'flatten_24calib')(maxpool_24calib)
    fc_24calib = Dense(units = 64, activation = 'relu', name = 'fc_24calib')(flatten_24calib)
    prediction_24calib = Dense(units = 45, activation = 'softmax', name = 'prediction_24calib')(fc_24calib)

    calib24 = keras.models.Model(inputs = input_24calib, outputs = prediction_24calib, name = 'calib24')

    return calib24

def create_calib48():
    input_48calib = Input(shape=(48,48,3), name = 'input_48calib')
    conv_1_48calib = Conv2D(filters=64, kernel_size=5, strides=1, input_shape = (48,48,3), activation = 'relu', name = 'conv_1_48calib')(input_48calib)
    maxpool_48calib = MaxPooling2D(pool_size = 3, strides=2, name = 'maxpool_48calib')(conv_1_48calib)
    norm_1_48calib = BatchNormalization(name = 'norm_1_48calib')(maxpool_48calib)
    conv_2_48calib = Conv2D(filters=64, kernel_size=5, strides=1, input_shape = (48,48,3), activation = 'relu', name = 'conv_2_48calib')(norm_1_48calib)
    flatten_48calib = Flatten(name = 'flatten_48calib')(conv_2_48calib)
    fc_48calib = Dense(units = 256, activation = 'relu', name = 'fc_48calib')(flatten_48calib)
    prediction_48calib = Dense(units = 45, activation = 'softmax', name = 'prediction_48calib')(fc_48calib)

    calib48 = keras.models.Model(inputs = input_48calib, outputs = prediction_48calib, name = 'calib48')

    return calib48

#--------------------------------------------- TRAINING ---------------------------------------------#

def train_calib(dim):
    train_img, train_label = load_calib_train(dim)

    print('Train start!')
    if int(dim) == 12:
        model = create_calib12()
        model_name = '12calib'
    elif int(dim) == 24:
        model = create_calib24()
        model_name = '24calib'
    elif int(dim) == 48:
        model = create_calib48()
        model_name = '48calib'

    model.compile(optimizer = keras.optimizers.SGD(learning_rate = learning_rate), 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])

    history = model.fit(train_img, train_label, 
            epochs = epoch_num,
            shuffle = True,
            batch_size = mini_batch_size,
            validation_split = val_split)
    
    model.save(model_path + model_name + ".h5")
    print("Saved model to disk")

    util.show_calib_train_history(history, model_name)

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])
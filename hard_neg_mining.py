import tensorflow as tf 
import keras
from keras.models import Model
from keras.models import load_model
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
from skimage.util.shape import view_as_windows
import os
import sys

import util

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --------------------------------------------- PARAMETERS --------------------------------------------- #

detection_model_path = 'detection/model/'
calibration_model_path = 'calibration/model/'

calib_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
calib_off_x = [-0.17, 0., 0.17]
calib_off_y = [-0.17, 0., 0.17]
calib_pattern_num = len(calib_scale) * len(calib_off_x) * len(calib_off_y)

window_stride = 2
face_minimum = 20
downscale = 0.8
pyramid_num = 10

threshold_12net = 0.5035
threshold_24net = 0.5045
threshold_48net = 0.505

# NMS
max_nms_12calib = 200
max_nms_24calib = 200
max_nms_48calib = 200
iou_12calib = 0.9
iou_24calib = 0.9
iou_48calib = 0.9


# ---------------------------------------- HARD NEGATIVE PIPELINE ---------------------------------------- #

def net12(input_12net):
    name12net = detection_model_path + '12net.h5'
    net12 = load_model(name12net,custom_objects={'recall': util.recall} )

    fc_12net_model = Model(
                    inputs = net12.input,
                    outputs = net12.get_layer(name = 'fc_12net').output)
    
    result_12net = [[] for i in range (0, len(input_12net))]
    negative_id = [[] for i in range (0, len(input_12net))]
    out_fc_12net = [[] for i in range (0, len(input_12net))]

    print('Run on 12 net.')

    for i in range (0, len(input_12net)):

        input_12net[i] = np.array(input_12net[i])

        # predict 12-net
        result_12net[i] = net12.predict(input_12net[i])
        predictions = util.predict_class(result_12net[i])

        negative_id[i] = np.where(predictions == 1)[0] #if positive

        if len(negative_id[i]) == 0:
            continue
        
        # predict 12-net fc
        out_fc_12net[i] = fc_12net_model.predict(input_12net[i]) # is it filtered too??

        result_12net[i] = tf.gather(result_12net[i], negative_id[i])
        result_12net[i] = np.array(result_12net[i])
        # print(result_12net[i])
        result_12net[i] = [f[1] for f in result_12net[i]] # positive score
        out_fc_12net[i] = tf.gather(out_fc_12net[i], negative_id[i])
        out_fc_12net[i] = np.array(out_fc_12net[i])

        print('positives 12: ', len(result_12net[i]))

    return result_12net, out_fc_12net, negative_id


def net24(input_24net, out_fc_12net):
    name24net = detection_model_path + '24net.h5'
    net24 = load_model(name24net,custom_objects={'recall': util.recall} )

    fc_24net_model = Model(
                    inputs = net24.input,
                    outputs = net24.get_layer(name = 'fc_1_24net').output)
    
    result_24net = [[] for i in range (0, len(input_24net))]
    negative_id = [[] for i in range (0, len(input_24net))]
    out_fc_24net = [[] for i in range (0, len(input_24net))]

    print('Run on 24 net.')

    for i in range(0, len(input_24net)):
        if len(input_24net[i]) == 0:
                continue

        input_24net[i] = np.array(input_24net[i])
        out_fc_12net[i] = np.array(out_fc_12net[i])

        # predict 24-net
        result_24net[i] = net24.predict({'input_24net': input_24net[i], 'input_from_12net': out_fc_12net[i]})
        predictions = util.predict_class(result_24net[i])
        negative_id[i] = np.where(predictions == 1)[0] #if positive

        # predict 24-net fc
        out_fc_24net[i] = fc_24net_model.predict({'input_24net': input_24net[i], 'input_from_12net': out_fc_12net[i]})

        result_24net[i] = tf.gather(result_24net[i], negative_id[i])
        result_24net[i] = np.array(result_24net[i])
        result_24net[i] = [f[1] for f in result_24net[i]] # positive score
        out_fc_24net[i] = tf.gather(out_fc_24net[i], negative_id[i])
        out_fc_24net[i] = np.array(out_fc_24net[i])
        out_fc_12net[i] = tf.gather(out_fc_12net[i], negative_id[i])
        out_fc_12net[i] = np.array(out_fc_12net[i])

        print('positives 24: ', len(result_24net[i]))
        
    return result_24net, out_fc_24net, out_fc_12net, negative_id

def net48(input_48net, out_fc_12net, out_fc_24net):
    name48net = detection_model_path + '48net.h5'
    net48 = load_model(name48net, custom_objects={'recall': util.recall})

    result_48net = [[] for i in range (0, len(input_48net))]
    negative_id = [[] for i in range (0, len(input_48net))]

    print('Run on 48 net.')

    for i in range(0, len(input_48net)):
        if len(input_48net[i]) == 0:
            continue

        input_48net[i] = np.array(input_48net[i])
        out_fc_24net[i] = np.array(out_fc_24net[i])
        out_fc_12net[i] = np.array(out_fc_12net[i])

        # predict 48-net
        result_48net[i] = net48.predict({
            'input_48net': input_48net[i], 'input48_from_24net': out_fc_24net[i], 'input48_from_12net': out_fc_12net[i]})
        predictions = util.predict_class(result_48net[i])
        negative_id[i] = np.where(predictions == 1)[0] #if positive

        result_48net[i] = tf.gather(result_48net[i], negative_id[i])
        result_48net[i] = np.array(result_48net[i])
        result_48net[i] = [f[1] for f in result_48net[i]] # positive score

        print('positives 48: ', len(result_48net[i]))
    
    return result_48net, negative_id


def calibnet(dim, input_calib, boxes, size):
    if dim == 12:
        name_calib = calibration_model_path + '12calib.h5'
        calib = load_model(name_calib)
        max = max_nms_12calib
        iou = iou_12calib
        print('Calibrate bounding boxes on 12 calib.')

    if dim == 24:
        name_calib = calibration_model_path + '24calib.h5'
        calib = load_model(name_calib)
        max = max_nms_24calib
        iou = iou_24calib
        print('Calibrate bounding boxes on 24 calib.')

    if dim == 48:
        name_calib = calibration_model_path + '48calib.h5'
        calib = load_model(name_calib)
        max = max_nms_48calib
        iou = iou_48calib
        print('Calibrate bounding boxes on 48 calib.')

        # CALIB START
        input_calib = np.array(input_calib)
        result_calib = calib.predict(input_calib)

        # CALIBRATE BOXES
        result_calib = np.array(result_calib)
        boxes = np.array(boxes)
        calib_boxes = util.calib_box(result_calib, boxes, size)

        return calib_boxes

    result_calib = [[] for i in range (0, len(input_calib))]
    calib_boxes = [[] for i in range (0, len(input_calib))]
    
    for i in range(0, len(input_calib)):
        if len(input_calib[i]) == 0:
                continue

        # CALIB START
        input_calib[i] = np.array(input_calib[i])
        result_calib[i] = calib.predict(input_calib[i])

        # CALIBRATE BOXES
        result_calib[i] = np.array(result_calib[i])
        boxes[i] = np.array(boxes[i])
        calib_boxes[i] = util.calib_box(result_calib[i], boxes[i], size)

    return calib_boxes


# -------------------------------------------------- RUNS -------------------------------------------------- #
nms_threshold = 0.9
nms_threshold_48 = 0.7

def run(dim):
    dim = int(dim)    
    neg_mining_path = 'dataset/negative mining/'

    mining_file_list = [f for f in os.listdir(neg_mining_path) if f.endswith('.jpg')]

    tot_iou = 0
    for i, name in enumerate(mining_file_list):

        input_nets, boxes, ori_img = util.load_test_data(name)
        input_nets = np.array(input_nets)
        height, width, channels = ori_img.shape # h = w

        # RESIZE FOR 12 NET
        input_12net = util.resize_data(12, input_nets)

        # 12 NET    
        scores12, out_fc_12net, negative_id_12 = net12(input_12net)
        boxes = util.gather_indices(boxes, negative_id_12)
        input_calib12 = util.gather_indices(input_12net, negative_id_12)
        input_nets = util.gather_indices(input_nets, negative_id_12)

        # 12 CALIB
        calib_boxes12 = calibnet(12, input_calib12, boxes, height)

        # NMS 12 CALIB
        nms_id = util.NMS(boxes, scores12, max_nms_12calib, iou_12calib, nms_threshold)
        calib_nms_12 = util.gather_indices(boxes, nms_id)

        input_nets = util.gather_indices(input_nets, nms_id)
        out_fc_12net = util.gather_indices(out_fc_12net, nms_id)

        if dim == 24:
            util.collect_hard_neg(input_nets, name, dim)

        if dim == 48:
            # RESIZE FOR 24 NET
            input_24net = util.resize_data(24, input_nets)

            # 24 NET
            scores24, out_fc_24net, out_fc_12net, negative_id_24 = net24(input_24net, out_fc_12net)
            boxes24 = util.gather_indices(calib_nms_12, negative_id_24)
            input_calib24 = util.gather_indices(input_24net, negative_id_24)
            input_nets = util.gather_indices(input_nets, negative_id_24)

            # 24 CALIB
            calib_boxes24 = calibnet(24, input_calib24, boxes24, height)

            # NMS 24 CALIB
            nms_id = util.NMS(boxes24, scores24, max_nms_24calib, iou_24calib, nms_threshold_48)
            calib_nms_24 = util.gather_indices(boxes24, nms_id)

            input_nets = util.gather_indices(input_nets, nms_id)

            util.collect_hard_neg(input_nets, name, dim)

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])
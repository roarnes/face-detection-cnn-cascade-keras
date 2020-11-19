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


train_path = '//Volumes/Seagate/SKRIPSI/calib_dataset/'
model_path = 'Model/'
history_path = 'History/'
original_path = '//Volumes/Seagate/SKRIPSI/LFW/lfw/'
test_path = 'Test/'
hd_dir = '//Volumes/Seagate/SKRIPSI/'
neg_path = '//Volumes/Seagate/SKRIPSI/COCO/hard_test/'

calib_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
calib_off_x = [-0.17, 0., 0.17]
calib_off_y = [-0.17, 0., 0.17]
calib_pattern_num = len(calib_scale) * len(calib_off_x) * len(calib_off_y)

window_stride = 2
face_minimum = 20
# downscale = float(12/face_minimum)
downscale = 0.8
upscale = 1.1
max_pyramid = 15

threshold_calib = 0.99

# NMS
max_nms_12calib = 20
max_nms_24calib = 20
max_nms_48calib = 1
iou_12calib = 0.5
iou_24calib = 0.5
iou_48calib = 0.5

global_resize = 100

def load_test_data(file): # load test data for 12net
    print('Loading test images..')
    path = 'dataset/negative mining/'

    file = path + file

    test_img = Image.open(file)
    test_img = test_img.resize((global_resize, global_resize), Image.BICUBIC)
    test_img_array = img_to_array(test_img)

    ori_img = cv2.imread(file) # unconverted image
    ori_img = cv2.resize(ori_img, (global_resize, global_resize), cv2.INTER_CUBIC)
    # ori_img = ori_img.copy()
    # h, w, c = ori_img.shape
    box_pyramid_sizes = create_box_pyramid(global_resize)
    input_nets, boxes = create_sliding_windows(test_img_array, box_pyramid_sizes)
    return input_nets, boxes, ori_img

def resize_data(dim, test_data):
    data = [[[] for j in range(0, len(test_data[i]))] for i in range (0, len(test_data))]
    for i in range(0, len(test_data)):
        test_data[i] = np.array(test_data[i])
        if len(test_data[i]) == 0:
            continue
        for j, image in enumerate(test_data[i]):
            image = array_to_img(image)
            image = image.resize((dim, dim), Image.BICUBIC)
            image = img_to_array(image)
            data[i][j] = image 
    return data

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def get_pyramid_num(size):
    num = 1
    bb = 12
    for i in range(1, max_pyramid):
        bb = int(bb/downscale)
        if bb >= size:
            break
        num += 1
    return num

def create_box_pyramid(size):
    pyramid_num = get_pyramid_num(size)

    box_pyramid_sizes = [[] for i in range (0, pyramid_num)]
    box_pyramid_sizes[0] = 12
    for i in range(1, pyramid_num):
        box_pyramid_sizes[i] = int(box_pyramid_sizes[i-1]/downscale) # upscale for box instead of downscale for image
    return box_pyramid_sizes

def create_sliding_windows(image, box_sizes):
    windows = [[] for i in range (0, len(box_sizes))]
    coords = [[] for i in range (0, len(box_sizes))]
    coord = []
    h, w, c = image.shape
    for i, box_dim in enumerate(box_sizes):
        window = view_as_windows(np.array(image), (box_dim, box_dim, 3), window_stride)
        window = [window[i][j][0] for i in range (0, len(window)) for j in range (0, len(window[0]))]
        coord = [[i, j, i+box_dim, j+box_dim] for i in range (0, w-box_dim+1, window_stride) for j in range (0, h-box_dim+1, window_stride)]
        coord = np.array(coord)
        window = np.array(window)
        windows[i] = window
        coords[i] = coord
    return windows, coords

# def collect_patterns(result):
#     for

def calib_box(result, boxes, size):
    calib_boxes = [[] for i in range (0, len(result))]
    patterns_id = [[] for i in range (0, len(result))]

    for i in range(0, len(result)):
        # pattern[i] = np.argmax(result[i])
        patterns_id[i] = np.where(result[i] >= threshold_calib)[0] #if positive

        if len(patterns_id[i]) == 0:
            calib_boxes[i] = boxes[i]
            continue # box is not calibrated
        
        s = 0
        x = 0
        y = 0

        # collect the values and get average
        for p in range(0, len(patterns_id[i])):
            s_i = patterns_id[i][p] // (len(calib_off_x) * len(calib_off_y))
            x_i = patterns_id[i][p] % (len(calib_off_x) * len(calib_off_y)) // len(calib_off_y)
            y_i = patterns_id[i][p] % (len(calib_off_x) * len(calib_off_y)) % len(calib_off_y) 
                    
            s += calib_scale[s_i]
            x += calib_off_x[x_i]
            y += calib_off_y[y_i]

        s = s / float(len(patterns_id[i]))
        x = x / float(len(patterns_id[i]))
        y = y / float(len(patterns_id[i]))

        new_ltx = boxes[i][0] + x * (boxes[i][2] - boxes[i][0])
        new_lty = boxes[i][1] + y * (boxes[i][3] - boxes[i][1])
        new_rbx = new_ltx + s * (boxes[i][2] - boxes[i][0])
        new_rby = new_lty + s * (boxes[i][3] - boxes[i][1])

        new_ltx = int(max(new_ltx,0))
        new_lty = int(max(new_lty,0))
        new_rbx = int(min(new_rbx, size-1))
        new_rby = int(min(new_rby, size-1))

        calib_boxes[i] = [new_ltx, new_lty, new_rbx, new_rby]

    return calib_boxes

def NMS(calib_boxes, scores, max, iou, score): # input 1d
    nms_id = [[] for i in range (0, len(calib_boxes))]
    nms_score = [[] for i in range (0, len(calib_boxes))]
    for i in range(0, len(calib_boxes)):
        if len(calib_boxes[i]) == 0 or len(scores[i]) == 0:
            continue
        calib_boxes[i] = np.array(calib_boxes[i])
        nms_id[i], nms_score[i] = tf.image.non_max_suppression_with_scores(
                    boxes = calib_boxes[i], 
                    scores = scores[i],
                    max_output_size = max, 
                    soft_nms_sigma = 0.5,
                    iou_threshold = iou,
                    score_threshold = score)
    
    return nms_id

def Global_NMS(calib_boxes, scores, max, iou):
    if len(calib_boxes) == 0:
        return []
    nms_id, nms_score = tf.image.non_max_suppression_with_scores(
                    boxes = calib_boxes, 
                    scores = scores,
                    max_output_size = max, 
                    soft_nms_sigma = 0.5,
                    iou_threshold = iou,
                    score_threshold = 0.9)
    return nms_id, nms_score

def resize_bounding_box(box, current_size, new_size):
    box = np.array(box)
    xmin = box[0]
    ymin = box[1]
    xmax = box[2] 
    ymax = box[3]
    # print(xmin, ymin, xmax, ymax)

    w = abs(xmax - xmin)
    h = abs(ymax - ymin)

    scale = new_size/current_size

    new_w = int(w * scale)
    new_h = int(h * scale)
    # print (scale, new_w, new_h)

    # step = new_w - w

    center_x = int((xmin + xmax) / 2 * scale)
    center_y = int((ymin + ymax) / 2 * scale)

    new_xmin = int(center_x - (new_w/2))
    new_ymin = int(center_y - (new_h/2))
    new_xmax = int(center_x + (new_w/2))
    new_ymax = int(center_y + (new_h/2))
    # print(new_xmin, new_ymin, new_xmax, new_ymax)

    return [new_xmin, new_ymin, new_xmax, new_ymax]

def predict_class(result):
    # print(result)
    class_prediction = [[] for i in range(0, len(result))]
    for i in range(0, len(result)):
        class_prediction[i] = [np.argmax(result[i])]
    class_prediction = np.array(class_prediction)
    return class_prediction

# def resize_image_from_calib(dim, boxes, images):
    # input_next_net = [[] for i in range(0, len(boxes))]
    # for i in range (0, len(boxes)):

    #     faces_next = []
    #     for ix in range (0, len(boxes[i])):
    #         xmin = boxes[ix][0]
    #         ymin = boxes[ix][1]
    #         xmax = boxes[ix][2] 
    #         ymax = boxes[ix][3]

    #         if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
    #             print('zeroooooo')
    #             continue

    #         faces_next.append(tf.image.convert_image_dtype(
    #                         cv2.resize(images[i][xmin:xmax,ymin:ymax], (dim, dim), cv2.INTER_CUBIC),
    #                         tf.float32))

    #     input_next_net[i] = faces_next
    
    # return input_next_net

def calculate_iou(test_box, i):
    if len(test_box) != 0:
        test_box = np.array(test_box)
        test_box = test_box[0]
        # ground_truth_box = [83,92,166,175]
        # ground_truth_box = resize_bounding_box(ground_truth_box, 250, global_resize)

        fp = open('facepoints.csv', 'r')
        test_file_list = fp.read().splitlines()

        x1 = [int(f.split(',')[1]) for f in test_file_list]
        y1 = [int(f.split(',')[2]) for f in test_file_list]
        x2 = [int(f.split(',')[3]) for f in test_file_list]
        y2 = [int(f.split(',')[4]) for f in test_file_list]
        ground_truth_box = [x1[i], y1[i], x1[i]+x2[i], y1[i]+y2[i]]

        #  determine the (x, y)-coordinates of the intersection rectangle
        xA = max(ground_truth_box[0], test_box[0])
        yA = max(ground_truth_box[1], test_box[1])
        xB = min(ground_truth_box[2], test_box[2])
        yB = min(ground_truth_box[3], test_box[3])

        # compute the area of intersection rectangle
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        gt_box_area = (ground_truth_box[2] - ground_truth_box[0] + 1) * (ground_truth_box[3] - ground_truth_box[1] + 1)
        test_box_area = (test_box[2] - test_box[0] + 1) * (test_box[3] - test_box[1] + 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(gt_box_area + test_box_area - inter_area)
        # return the intersection over union value
        return iou
    return 0

def gather_indices(array, indices):
    trim = []
    for i in range (0, len(array)):
        if len(array[i]) == 0:
            continue
        if indices[i] == []:
            array[i] = []
            continue
        array[i] = np.array(array[i])
        indices[i] = np.array(indices[i])
        array[i] = tf.gather(array[i], indices[i])

    # for i in range(0, len(trim)):
    #     del array[trim[i]]
    
    return array

def draw_pipeline(boxes, image, dim, name, i):
    # path = '//Volumes/Seagate/SKRIPSI/Face Recognition/Raw/'
    f = name.split('/')
    filename = f[-1]
    foldername = f[0]

    # save_dir = '//Volumes/Seagate/SKRIPSI/Face Recognition/Faces/Train/' + foldername +'/'
    save_dir = '//Volumes/Seagate/SKRIPSI/Face Recognition/Faces/Test/' + foldername +'/'

    # f = name.split('/')[-1]
    # f = f.split('.')[0]
    # save_dir2 = hd_dir + 'Pipeline/' + foldername + '_' + str(i)
    save_dir2 = hd_dir + 'Pipeline_Test/' + foldername + '_' + str(i)
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    h, w, c = image.shape
    if dim == 48:
        for ix in range (0, len(boxes)):
            draw_ori = image.copy()
            xmin = boxes[ix][0]
            ymin = boxes[ix][1]
            xmax = boxes[ix][2] 
            ymax = boxes[ix][3]

            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                continue

            draw_ori = cv2.rectangle(draw_ori, (xmin, ymin), (xmax, ymax), (255), 2)

            fp = open('facepoints.csv', 'r')
            test_file_list = fp.read().splitlines()

            x1 = [int(f.split(',')[1]) for f in test_file_list]
            y1 = [int(f.split(',')[2]) for f in test_file_list]
            x2 = [int(f.split(',')[3]) for f in test_file_list]
            y2 = [int(f.split(',')[4]) for f in test_file_list]

            ground_truth_box = [x1[i], y1[i], x1[i]+x2[i], y1[i]+y2[i]]
            # print(y1[i], x1[i], y2[i], x2[i])

            # ground_truth_box = [83,92,166,175]
            # ground_truth_box = resize_bounding_box(ground_truth_box, 100, h)

            top_left = (ground_truth_box[0], ground_truth_box[1])
            bottom_right = (ground_truth_box[2], ground_truth_box[3])

            cv2.rectangle(draw_ori, top_left, bottom_right, (0,0,255), 2)

            cv2.imwrite(save_dir2 + '/after_' + str(dim) + '_calib_' + str(ix) + '.jpg', draw_ori)
            cv2.imwrite(save_dir + filename, image[ymin:ymax, xmin:xmax])

    else:
        for i in range (0, len(boxes)):
            if len(boxes[i]) == 0:
                continue
            draw_ori = image.copy()
            for ix in range (0, len(boxes[i])):
                xmin = boxes[i][ix][0]
                ymin = boxes[i][ix][1]
                xmax = boxes[i][ix][2] 
                ymax = boxes[i][ix][3]

                if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                    continue

                draw_ori = cv2.rectangle(draw_ori, (xmin, ymin), (xmax, ymax), 255, 2)

            cv2.imwrite(save_dir2 + '/after_' + str(dim) + '_calib_' + str(i) + '.jpg', draw_ori)

def collect_hard_neg(input_net, filename, dim):
    hard_neg_dir = 'dataset/hard negative/'
    save_hard_neg = hard_neg_dir + str(dim) + '/'
    # filename = filename.split('/')[-1]
    filename = filename.split('.')[0]

    for i in range(0, len(input_net)):
        input_net[i] = np.array(input_net[i])
        for ix in range (0, len(input_net[i])):
            img = input_net[i][ix]
            img = array_to_img(img)
            # img = img.resize((dim, dim), Image.BICUBIC)
            img.save(save_hard_neg + filename + '_' + str(i) + str(ix) + '.jpg')
            print(save_hard_neg + filename + '_' + str(i) + str(ix) + '.jpg')

def show_calib_train_history(history, model_name):
    history_path = 'calibration/history/'

    model_name = history_path + model_name
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + '_accuracy.png', dpi = 100)
    print('Saved accuracy graph of model ' + model_name)
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.savefig(model_name + '_loss.png', dpi = 100)
    print('Saved loss graph of model ' + model_name)
    plt.show()

    print('Saving overall history of model ' + model_name)
    np.save(model_name + '.npy', history.history)

def show_detect_train_history(history, model_name):
    history_path = 'detection/history/'

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_path + model_name + '_accuracy.png', dpi = 100)
    print('Saved accuracy graph of model ' + model_name)
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_path + model_name + '_loss.png', dpi = 100)
    print('Saved loss graph of model ' + model_name)
    plt.show()

    # Plot training & validation recall values
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_path + model_name + '_recall.png', dpi = 100)
    print('Saved recall graph of model ' + model_name)
    plt.show()

    print('Saving overall history of model ' + model_name)
    np.save(history_path + model_name + '.npy', history.history)

    # To load the history:
    #history = np.load('my_history.npy',allow_pickle='TRUE').item()
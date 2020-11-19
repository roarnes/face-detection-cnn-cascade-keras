import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import sys


# list of all images in the LFW dataset in file_names.txt
file_names = 'file_names.txt'

lfw_path = '//Volumes/Seagate/SKRIPSI PROJECT/calib_dataset/Original/'
# lfw_path = 'dataset/LFW/'

faces_path = 'dataset/faces/'
calib_path = 'dataset/calib/'

# crop image from the original LFW dataset to get only the faces (without background)
def detection_faces():
    save_path = faces_path

    ori_fp = open(file_names, 'r')
    file_list = ori_fp.read().splitlines()

    for i in range(0, len(file_list[:2])):
        file_name = file_list[i].split('/')[-1]
        ori_img = cv2.imread(lfw_path + file_list[i])

        # upper-left and lower-right corners being (83,92) and (166,175), respectively. 
        x1,y1 = (83,92)
        x2,y2 = (166,175)

        cropped_img = ori_img[y1:y2 , x1:x2]
        cv2.imwrite(save_path + file_name, cropped_img)

        print('Saved image to', save_path + file_name)

def calibration_faces():
    
    ori_fp = open(file_names, 'r')
    file_list = ori_fp.read().splitlines()

    print('Found ', len(file_list), ' images.')
    print('Loading calib data...')

    calib_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
    calib_off_x = [-0.17, 0., 0.17]
    calib_off_y = [-0.17, 0., 0.17]
    calib_pattern_num = len(calib_scale) * len(calib_off_x) * len(calib_off_y)

    # faces in all images are in this bounding area
    xmin, ymin = (83,92)
    xmax, ymax = (166,175)

    train_image = []
    train_label = []

    for i, image_path in enumerate(file_list[:2]):
        file_name = image_path.split('/')[-1]
        print('Processing ', file_name)
        img = Image.open(lfw_path + image_path)

        if xmax >= img.size[0]:
            xmax = img.size[0]-1
        if ymax >= img.size[1]:
            ymax = img.size[1]-1
        
        calib_list = [0 for _ in range(calib_pattern_num)]

        for si, s in enumerate(calib_scale):         
            for xi, x in enumerate(calib_off_x):
                for yi, y in enumerate(calib_off_y):

                    label_list = [0 for _ in range(calib_pattern_num)]
                    
                    new_xmin = xmin - x * float(xmax-xmin)/s
                    new_ymin = ymin - y * float(ymax-ymin)/s
                    new_xmax = new_xmin + float(xmax-xmin)/s
                    new_ymax = new_ymin + float(ymax-ymin)/s
                    
                    new_xmin = int(new_xmin)
                    new_ymin = int(new_ymin)
                    new_xmax = int(new_xmax)
                    new_ymax = int(new_ymax)

                    if new_xmin < 0 or new_ymin < 0 or new_xmax >= img.size[0] or new_ymax >= img.size[1]:
                        continue

                    cropped_img = img.crop((new_xmin, new_ymin, new_xmax, new_ymax))
                    pattern_id = si * len(calib_off_x) * len(calib_off_y) + xi * len(calib_off_y) + yi

                    save_path = calib_path + str(pattern_id) + '/'
                    
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    cropped_img.save(save_path + file_name, 'jpeg')
                    print('Saved image to', save_path + file_name)

def nonfaces():
    dir = '//Volumes/Seagate/SKRIPSI PROJECT/COCO/Unused COCO/'
    save_dir = 'dataset/non-faces/'
    save_dir_for_mining = 'dataset/negative mining/'
    files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
    files = files[12000:12005]

    new_width = 400
    new_height = 400

    for image in files:
        fname = image.split('.')[0]
        print(image)
        im = Image.open(dir + image)
        width, height = im.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        # For training detection nets
        im = im.crop((left, top, right, bottom))
        im.save(save_dir + fname + '_1.jpg')

        # Crop starting from top left
        # For hard negative mining
        im = im.crop((0, 0, 200, 200))
        im.save(save_dir_for_mining + fname + '_2.jpg')

if __name__ == '__main__':
    globals()[sys.argv[1]]()
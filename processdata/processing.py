# coding: utf-8
import cv2
from math import *
import math
import numpy as np
import os
import glob
import random
import shutil
import PIL
from PIL import Image

def split_train():
    train_percent = 0.9
    val_percent = 0.1

    total_img = sorted(glob.glob(os.path.join(IMGPAYH, '*.jpg')))
    total_txt = sorted(glob.glob(os.path.join(TXTPATH, '*.txt')))
    # 获取列表的总数
    num = len(total_img)
    list = range(num)

    tv = int(num * val_percent)
    trainval = random.sample(list, tv)
    for i in range(len(trainval)):
        print('move %s'% total_img[trainval[i]])
        shutil.move(total_img[trainval[i]], NEWIMGPATH)
        shutil.move(total_txt[trainval[i]], NEWTXTPATH)
    print('success')




IMGPAYH = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/mtwi_2018_train/image_train/'
TXTPATH = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/mtwi_2018_train/txt_train/'

NEWIMGPATH = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/mtwi_2018_train/train_mtwi_img/'
NEWTXTPATH = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/mtwi_2018_train/train_mtwi_txt/'

imagelist = sorted(glob.glob(os.path.join(IMGPAYH, '*.jpg')))
txtlist = sorted(glob.glob(os.path.join(TXTPATH, '*.txt')))
if __name__ == '__main__':
    # split_train()
    num = 0
    for i in range(len(imagelist)):
        print('number: %s image' % i)
        name = "img_{:0>5d}".format(i)
        new_img = NEWIMGPATH + name + '.jpg'
        new_txt = NEWTXTPATH + name + '.txt'
        F_write = open(new_txt, 'w')
        image = Image.open(imagelist[i]).convert("RGB")
        print(image.format)
        image.save(new_img, quality=95)
        if image is None:
            num += 1
        # draw = image.copy()
        F_read = open(txtlist[i], 'rb')
        lines = F_read.readlines()  # 逐行读入内容
        for line in lines:
            write_line = str(line, encoding="utf-8")
            F_write.write(write_line)
            # position = write_line.split(',')
            # left_up = (int(float(position[0])), int(float(position[1])))
            # left_down = (int(float(position[2])), int(float(position[3])))
            # right_up = (int(float(position[6])), int(float(position[7])))
            # right_down = (int(float(position[4])), int(float(position[5])))
            # cv2.rectangle(draw, left, right, color=(255, 0, 0), thickness=2)
            # cv2.line(draw, left_up, left_down, color=(255, 0, 0),thickness=2)
            # cv2.line(draw, left_down, right_down, color=(255, 0, 0), thickness=2)
            # cv2.line(draw, right_down, right_up, color=(255, 0, 0), thickness=2)
            # cv2.line(draw, left_up, right_up, color=(255, 0, 0), thickness=2)
        F_write.close()
        F_read.close()
        # cv2.imshow("draw", draw)
        # cv2.imwrite(new_img, image)
        # image.save(new_img, optimize=True)
        # cv2.waitKey(0)
    print(num)
    print('success')

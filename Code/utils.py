#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os
import re
from PIL import Image
import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


def read_directory(image_set: str) -> list:
    return [os.path.join(image_set,  f) for f in sorted(os.listdir(image_set), key=lambda x:int(re.sub("\D","",x)))]

def load_image(path):
    # img = Image.open(path)
    img = cv2.imread(path)
    return img

def preprocess_image(cv_img):
    img = np.array(cv_img)
    # img = (img - 127.5) / 127.5
    # img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = img / 255.0

    return img

def deprocess_image(img):
    # img = img * 127.5 + 127.5
    img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    # img = remap(img, 0, 255, 0.0, 1.0)

    return img

def data_augmentation(img, probability=25, random_horizontal_flip=False, random_verticle_flip=False, random_rotation=False, random_transpose=False):
    
    if(random_horizontal_flip and np.random.randint(100) > (100-probability)):
        return np.flip(img, 1)

    if(random_verticle_flip and np.random.randint(100) > (100-probability)):
        return np.flip(img, 1)

    return img

def get_image(Image, PreprocessImage=True, Resize=(1024,256)):
    if(PreprocessImage):
        # print("preprocess_image")
        x = preprocess_image(Image)
    else:
        # print("no preprocess_image")
        x = Image

    if(Resize is not None):
        x = cv2.resize(x, Resize, interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("", deprocess_image(x))
    # cv2.waitKey()

    return x

def save_image(np_arr, path):
    img = deprocess_image(np_arr)
    im = Image.fromarray((img).astype(np.uint8))
    im.save(path)

def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

     # portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result
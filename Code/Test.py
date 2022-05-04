import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Config as cf
import DataGenerator as dg


def predict(model, rgb_cv_img):
    rgb_cv_img = cv2.resize(rgb_cv_img, (cf.WIDTH, cf.HEIGHT))
    input_img = np.reshape(rgb_cv_img, (1, cf.HEIGHT, cf.WIDTH, 3))

    tf_img = tf.image.convert_image_dtype(input_img, tf.float32)

    pred = model(tf_img)
    pred = np.squeeze(pred)

    return pred


def evaluate(test_dir, random_test=False):

    model = tf.keras.models.load_model(os.path.join(cf.MODEL_DIR, cf.MODEL_CHKPT_NAME))

    rgb_images, seg_images = dg.load_data(test_dir)

    cv2.namedWindow("input", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("input", 50, 200)
    cv2.namedWindow("pred", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("pred", 500, 200)
    cv2.namedWindow("true", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("true", 1000, 200)

    if random_test:
        while True:
            random_idx = random.randint(0, len(rgb_images) - 1)
            print("[INFO]: Testing on ", rgb_images[random_idx])
            rgb_img = cv2.imread(rgb_images[random_idx])

            pred = predict(model, rgb_img)
            true = cv2.imread(seg_images[random_idx])

            cv2.imshow("input", rgb_img)
            cv2.imshow("pred", pred)
            cv2.imshow("true", true)
            cv2.waitKey(0)
    else:
        for rgb_path, seg_path in zip(rgb_images, seg_images):
            print("[INFO]: Testing on ", rgb_path)
            rgb_img = cv2.imread(rgb_path)

            pred = predict(model, rgb_img)
            true = cv2.imread(seg_path)

            cv2.imshow("input", rgb_img)
            cv2.imshow("pred", pred)
            cv2.imshow("true", true)
            cv2.waitKey(0)


def test(input_dir):

    model = tf.keras.models.load_model(os.path.join(cf.MODEL_DIR, cf.MODEL_CHKPT_NAME))

    cv2.namedWindow("input", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("input", 50, 200)
    cv2.namedWindow("pred", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("pred", 500, 200)

    for root, di, files in os.walk(input_dir):
        for filename in files:
            rgb_path = os.path.join(root, filename)
            print("[INFO]: Testing on ", rgb_path)
            rgb_img = cv2.imread(rgb_path)

            rgb_img = cv2.resize(rgb_img, (cf.WIDTH, cf.HEIGHT))

            pred = predict(model, rgb_img)

            plt.imshow(pred)
            plt.show()
            # cv2.imshow("input", rgb_img)
            # cv2.imshow("pred", pred)
            # cv2.waitKey(0)


if __name__ == "__main__":
    # test_dir = "../Data/Train1"
    # evaluate(test_dir, random_test=True)

    input_dir = "../Data/Test"
    test(input_dir)

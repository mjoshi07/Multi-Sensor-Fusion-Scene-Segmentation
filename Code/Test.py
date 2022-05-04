#!/usr/env/bin python3

"""
CMSC733 Spring 2021: Classical and Deep Learning Approaches for Geometric Computer Vision
Homework 0: Alohomora: Phase 2


Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing modules
import time
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import sys
import argparse
import cv2
from PIL import Image
import pandas as pd
import traceback
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from Network.Network import CIFAR10Model, ResNet20, ResNeXt, DenseNet
from Misc.Helper import *
from Misc.MiscUtils import *

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class Test():

    def __init__(self, Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath, PreprocessImage=True):
        self.DatasetPath = DatasetPath
        self.LabelsFilePath = LabelsFilePath
        self.BatchSize = BatchSize
        self.CheckpointPath = CheckpointPath
        self.PreprocessImage = PreprocessImage

        self.Model = Model
        self.Loss = CategoricalCrossentropy()
        self.Accuracy = CategoricalAccuracy()
        self.TestingData = None
        self.Labels = None
        self.y_pred = np.empty([0])
        self.y_true = np.empty([0])
        self.Classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']

    def evaluate(self):

        running_loss = 0.0
        testing_accuracy = 0.0
        inference_time = 0.0

        for batch in tqdm(range(int(len(self.TestingData)/self.BatchSize))):
            X = np.empty([0, 32, 32, 3]) # Input to the network
            y = np.empty([0, 10]) # Ground Truth

            for index in range(self.BatchSize):
                X = np.insert(X, index, get_image(load_image(self.TestingData[(batch*self.BatchSize)+index]), self.PreprocessImage), axis=0)
                y = np.insert(y, index, one_hot_encoding(int(self.Labels[(batch*self.BatchSize)+index]), 10), axis=0)

            tick = time.time()

            y_ = self.Model(X, training=False) # Prediction of the network
            
            toc = time.time()
            inference_time += (toc - tick)/60

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)) # Cross entropy loss

            # Accuracy
            y_pred = tf.argmax(y_, axis=1)
            y_true = tf.argmax(y, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.math.equal(y_pred, y_true), dtype=tf.float32))

            self.y_pred = np.insert(self.y_pred, len(self.y_pred), y_pred, axis=0)
            self.y_true = np.insert(self.y_true, len(self.y_true), y_true, axis=0)

            running_loss += loss.numpy()
            testing_accuracy += accuracy

        print((toc - tick)/60)
        # result = generateResult(self.TestingData[(batch*self.BatchSize)+index], deprocess_H4_data(y_.numpy()), self.H_AB_training_data[batch], [240, 320], [128,128], 32, False)
        # cv2.imwrite(os.path.join(self.ResultsPath, 'Train', '%d.png'%self.Epoch), result)
        running_loss = running_loss/(int(len(self.TestingData)/self.BatchSize)) # Mean loss of the epoch
        print("Testing Loss: %f"%running_loss)
        testing_accuracy = testing_accuracy/(int(len(self.TestingData)/self.BatchSize))
        print("Testing Accuracy: %f"%testing_accuracy)
        print("Average Inference Time: %f"%(inference_time/int(len(self.TestingData)/self.BatchSize)))

        return running_loss, testing_accuracy

    def test(self):

        print("GPU %d"%tf.test.is_gpu_available()) # Checking GPU availability

        self.Model.summary() # Summary of the model

        if(self.CheckpointPath):
            self.TrainingDir = os.path.join('../Training', self.CheckpointPath) # Initializing the training directory for current training instance
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)
        else:
            self.TrainingDir = os.path.join(self.TrainingDir, datetime.now().strftime("%Y%m%d-%H%M%S")) # Initializing the training directory for current training instance
            os.makedirs(self.TrainingDir)
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            os.makedirs(self.CheckpointPath)
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            os.makedirs(self.ResultsPath)
            os.makedirs(os.path.join(self.ResultsPath, 'Test'))
            os.makedirs(os.path.join(self.ResultsPath, 'Val'))
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)

        learning_rate = 1E-3 # Learning rate of 0.005 from the paper
        self.Optimizer = Adam(learning_rate=learning_rate)

        # Creating a Checkpoint Manager
        ckpt = Checkpoint(step=tf.Variable(0), model=self.Model, optimizer=self.Optimizer)
        ckpt_manager = CheckpointManager(ckpt, self.CheckpointPath, max_to_keep=5)

        # Loading Traning Data
        self.TestingData = readImageSet(os.path.join(self.DatasetPath, 'Test')) # Getting relative paths to testing data
        self.Labels = readLabels(self.LabelsFilePath)

        ckpt.restore(ckpt_manager.latest_checkpoint) # Restoring checkpoint if available
        if(ckpt_manager.latest_checkpoint):
            self.Model = ckpt.model # Restoring model state
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
            self.Optimizer.lr.assign(ckpt.optimizer.lr) # Restoring learning rate of the optimizer
            print("Restored Learnign Rate: {}".format(self.Optimizer.lr))
        else:
            print("Initializing from scratch.")

        start = time.time() # Training instance start time

        try:
            testing_epoch_loss, testing_accuracy = self.evaluate() # Test the model

            end = time.time()

            cf_matrix = confusion_matrix(self.y_true, self.y_pred)

            fig = plt.figure()
            ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

            # ax.set_title('Confusion Matrix\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            ax.xaxis.set_ticklabels(['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'])
            ax.yaxis.set_ticklabels(['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'])

            fig.set_size_inches(10, 9)
            plt.savefig(os.path.join(self.ResultsPath, self.Model.name + ' - Testing - Cofusion Matrix.png'), dpi=100,  pad_inches=0)
            # plt.show()

            print("Took %.03f minutes to test"%((end-start)/60))
        except KeyboardInterrupt:
            print(traceback.format_exc())

            end = time.time()

            cf_matrix = confusion_matrix(self.y_true, self.y_pred)

            fig = plt.figure()
            ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

            # ax.set_title('Confusion Matrix\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            ax.xaxis.set_ticklabels(['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'])
            ax.yaxis.set_ticklabels(['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'])

            fig.set_size_inches(9, 9)
            plt.savefig(os.path.join(self.ResultsPath,  self.Model.name + ' - Testing - Cofusion Matrix.png'), dpi=100,  pad_inches=0)
            # plt.show()

            print("Took %.03f minutes to test"%((end-start)/60))


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', type=str, default="CIFAR10-Vanilla", help='Select model', choices=('CIFAR10-Vanilla', 'CIFAR10-Improved', 'ResNet', 'ResNeXt', 'DenseNet'))
    Parser.add_argument('--DatasetPath', type=str, default="../CIFAR10", help='Path to the testing data')
    Parser.add_argument('--LabelsFilePath', type=str, default="./TxtFiles/LabelsTest.txt", help='Path to the LabelsTrain.txt file')
    Parser.add_argument('--BatchSize', type=int, default=1, help='batch_size')
    Parser.add_argument('--CheckpointPath', type=str, default=None, help='Checkpoint for inference/resuming training')

    Args = Parser.parse_args()
    ModelType = Args.ModelType # Model to test
    DatasetPath = Args.DatasetPath # Path to the dataset folder containing 'Train' and 'Val' folders
    LabelsFilePath = Args.LabelsFilePath
    BatchSize = Args.BatchSize # Batchsize to be used in training
    CheckpointPath = Args.CheckpointPath # Folder name of trianing instance in the TrainingDir 

    if(ModelType == "CIFAR10-Vanilla"):
        Model = CIFAR10Model(BN=False)
        test_model = Test(Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath, PreprocessImage=False)
        test_model.test()

    if(ModelType == "CIFAR10-Improved"):
        Model = CIFAR10Model()
        test_model = Test(Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath)
        test_model.test()

    if(ModelType == "ResNet"):
        Model = ResNet20()
        test_model = Test(Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath)
        test_model.test()

    if(ModelType == "ResNeXt"):
        Model = ResNeXt(Depth=11, Cardinality=4, BottleneckWidth=24)
        test_model = Test(Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath)
        test_model.test()

    if(ModelType == "DenseNet"):
        Model = DenseNet()
        test_model = Test(Model, DatasetPath, LabelsFilePath, BatchSize, CheckpointPath)
        test_model.test()

if __name__ == '__main__':
    main()
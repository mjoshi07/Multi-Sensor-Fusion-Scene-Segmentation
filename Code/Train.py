#!/usr/env/bin python3

"""
ENPM673 Spring 2022: Classical and Deep Learning Approaches for Geometric Computer Vision
Project 4


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
from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy

from UNet import *
from utils import *

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class Train():

    def __init__(self, Model, DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath=None, CheckpointEpoch=None, Random=False, PreprocessImage=True, DataAugmentation=False, LR_Decay=True, DisableLogData=True):
        self.DatasetPath = DatasetPath
        self.Epochs = Epochs + 1
        self.BatchSize = BatchSize
        self.TrainingDir = TrainingDir
        self.CheckpointPath = CheckpointPath
        self.CheckpointEpoch = CheckpointEpoch
        self.Random = Random
        self.PreprocessImage = PreprocessImage
        self.DataAugmentation = DataAugmentation
        self.LR_Decay = LR_Decay
        self.DisableLogData = DisableLogData

        self.Model = Model
        self.Optimizer = None
        self.Epoch = 0 # Current epoch of the training operation
        self.Loss = MeanSquaredError()
        # self.Accuracy = CategoricalAccuracy()
        self.TrainingData = None
        self.ValidationData = None
        self.TrainingLabels = None
        self.ValidationLabels = None
        self.y_pred = np.empty([0])
        self.y_true = np.empty([0])

    def fit(self):

        running_loss = 0.0
        training_accuracy = 0.0
        self.y_pred = np.empty([0])
        self.y_true = np.empty([0])

        for batch in tqdm(range(int(len(self.TrainingData)/self.BatchSize))):
            X = np.empty([0, 256, 1024, 3]) # Input to the network
            y = np.empty([0, 256, 1024, 3]) # Ground Truth

            for index in range(self.BatchSize):

                if(self.Random):
                    idx = np.random.randint(len(self.TrainingData))
                    X = np.insert(X, index, get_image(load_image(self.TrainingData[idx]), self.PreprocessImage), axis=0)
                    y = np.insert(y, index, get_image(load_image(self.TrainingLabels[idx]), self.PreprocessImage), axis=0)
                else:
                    X = np.insert(X, index, get_image(load_image(self.TrainingData[(batch*self.BatchSize)+index]), self.PreprocessImage), axis=0)
                    y = np.insert(y, index, get_image(load_image(self.TrainingLabels[(batch*self.BatchSize)+index]), self.PreprocessImage), axis=0)

            with tf.GradientTape() as tape:
                y_ = self.Model(X, training=True) # Prediction of the network
                loss = self.Loss(y, y_)

            grads = tape.gradient(loss, self.Model.trainable_variables) # Calculating gradients
            self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_variables)) # Updating model weights
            
            running_loss += loss.numpy()

        save_image(np.vstack((np.vstack((X[-1], y_[-1])), y[-1])), os.path.join(self.ResultsPath, 'Train', str(self.Epoch) + '.png'))
        running_loss = running_loss/(int(len(self.TrainingData)/self.BatchSize)) # Mean loss of the epoch
        print("Training Loss: %f"%running_loss)
        training_accuracy = 0

        return running_loss, training_accuracy

    def validate(self):

        running_loss = 0.0
        validation_accuracy = 0.0

        for batch in tqdm(range(int(len(self.ValidationData)/self.BatchSize))):
            X = np.empty([0, 256, 1024, 3]) # Input to the network
            y = np.empty([0, 256, 1024, 3]) # Ground Truth

            for index in range(self.BatchSize):
                if(self.Random):
                    idx = np.random.randint(len(self.ValidationData))
                    X = np.insert(X, index, get_image(load_image(self.ValidationData[idx]), self.PreprocessImage), axis=0)
                    y = np.insert(y, index, get_image(load_image(self.ValidationLabels[idx]), self.PreprocessImage), axis=0)
                else:
                    X = np.insert(X, index, get_image(load_image(self.ValidationData[(batch*self.BatchSize)+index]), self.PreprocessImage), axis=0)
                    y = np.insert(y, index, get_image(load_image(self.ValidationLabels[(batch*self.BatchSize)+index]), self.PreprocessImage), axis=0)

            y_ = self.Model(X, training=False) # Prediction of the network
            loss = self.Loss(y, y_)

            running_loss += loss.numpy()

        save_image(np.vstack((np.vstack((X[-1], y_[-1])), y[-1])), os.path.join(self.ResultsPath, 'Val', str(self.Epoch) + '.png'))
        running_loss = running_loss/(int(len(self.ValidationData)/self.BatchSize)) # Mean loss of the epoch
        print("Validation Loss: %f"%running_loss)
        validation_accuracy = 0

        return running_loss, validation_accuracy

    def lr_scheduler(self, epoch):
        """
            Learning Rate Scheduler as given in the book - Advanced Deep Learning with TensorFlow 2 and Keras: Apply DL, GANs, VAEs, deep RL, unsupervised learning, object detection and segmentation, and more
        """

        lr = 1E-3
        if epoch > 180:
            lr *= 0.5E-3
        elif epoch > 160:
            lr *= 1E-3
        elif epoch > 120:
            lr *= 1E-2
        elif epoch > 80:
            lr *= 1E-1

        return lr

    def train(self):

        print("GPU %d"%tf.test.is_gpu_available()) # Checking GPU availability

        self.Model.summary() # Summary of the model

        print("Training Parameters -")
        print("Preprocess Data: {}".format(self.PreprocessImage))
        print("Data Augmentation: {}".format(self.DataAugmentation))

        if(self.CheckpointPath):
            self.TrainingDir = os.path.join(self.CheckpointPath) # Initializing the training directory for current training instance
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)

            StartEpoch = self.CheckpointEpoch + 1 # Initializing starting epoch of the training operation
        else:
            self.TrainingDir = os.path.join(self.TrainingDir, self.Model.name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")) # Initializing the training directory for current training instance
            os.makedirs(self.TrainingDir)
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            os.makedirs(self.CheckpointPath)
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            os.makedirs(self.ResultsPath)
            os.makedirs(os.path.join(self.ResultsPath, 'Train'))
            os.makedirs(os.path.join(self.ResultsPath, 'Val'))
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)

            StartEpoch = 1 # Initializing starting epoch of the training operation

        learning_rate = 1E-3 # Learning rate for the Adam optimizier
        self.Optimizer = Adam(learning_rate=learning_rate) # Initializing the Adam optimizer

        # Creating a Checkpoint Manager
        ckpt = Checkpoint(step=tf.Variable(0), model=self.Model, optimizer=self.Optimizer)
        ckpt_manager = CheckpointManager(ckpt, self.CheckpointPath, max_to_keep=None)

        # Loading Traning Data
        self.TrainingData, self.ValidationData, self.TrainingLabels, self.ValidationLabels = train_test_split(read_directory(os.path.join(self.DatasetPath, 'Train')), read_directory(os.path.join(self.DatasetPath, 'GT')), test_size=0.2, shuffle=False)

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
            for epoch in range(StartEpoch, self.Epochs):
                print("Epoch %d of %d"%((epoch), self.Epochs-1))
                self.Epoch = epoch # Updating current epoch


                if(self.LR_Decay):
                    # if((epoch)%20 == 0):
                        # learning_rate = learning_rate / 2
                        # print("Updated Learning Rate: %f"%learning_rate)
                        # self.Optimizer.lr.assign(learning_rate)
                    lr = self.lr_scheduler(epoch)
                    if(self.Optimizer.lr != lr):
                        print("Updated Learning Rate: %f"%lr)
                        self.Optimizer.lr.assign(lr)

                training_epoch_loss, training_accuracy = self.fit() # Training
                validation_epoch_loss, validation_accuracy = self.validate() # Validation

                ckpt.step.assign_add(1)
                if(epoch%10 == 0):
                    save_path = ckpt_manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                if(self.DisableLogData):
                    data = [epoch, np.mean(self.Optimizer.lr.numpy()), np.mean(training_epoch_loss), np.mean(validation_epoch_loss)]
                    if(epoch == 1):
                        df = pd.DataFrame([data], columns = ['Epochs', 'Learning Rate','Training Loss', 'Validation Loss'])
                        df.to_csv(os.path.join(self.TrainingDir, 'Training.csv'), mode='a')
                    else:
                        df = pd.DataFrame([data])
                        df.to_csv(os.path.join(self.TrainingDir, 'Training.csv'), header=False, mode='a')

                self.Epoch += 1

            end = time.time()
            print("Took %.03f minutes to train"%((end-start)/60))
        except KeyboardInterrupt:
            print(traceback.format_exc())
            save_path = ckpt_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            end = time.time()
            print("Took %.03f minutes to train"%((end-start)/60))


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DatasetPath', type=str, default="../Data/KITTI", help='Path to the training data')
    Parser.add_argument('--Epochs', type=int, default=50, help='Epochs to train the model for')
    Parser.add_argument('--BatchSize', type=int, default=8, help='Mini-batch_size')
    Parser.add_argument('--TrainingDir', type=str, default="../Training/", help='Path to save the training and validation results')
    Parser.add_argument('--CheckpointPath', type=str, default=None, help='Checkpoint for inference/resuming training')
    Parser.add_argument('--CheckpointEpoch', type=int, default=0, help='Checkpoint epoch to resume training from')
    Parser.add_argument('--DisableLogData', action='store_false', help='Toggle for logging data')

    Args = Parser.parse_args()
    DatasetPath = Args.DatasetPath # Path to the dataset folder containing 'Train' and 'Val' folders
    Epochs = Args.Epochs # Total number of epochs to run the training for
    BatchSize = Args.BatchSize # Batchsize to be used in training
    TrainingDir = Args.TrainingDir # Path to the directory where training data has to be stored
    CheckpointPath = Args.CheckpointPath # Folder name of trianing instance in the TrainingDir 
    CheckpointEpoch = Args.CheckpointEpoch # Number of epochs for which training has been completed (will resume from CheckpointEpoch + 1)
    DisableLogData = Args.DisableLogData # Toggle for logging data

    Model = UNet()
    train_model = Train(Model, DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath, CheckpointEpoch, Random=False, DisableLogData=DisableLogData)
    train_model.train()

if __name__ == '__main__':
    main()
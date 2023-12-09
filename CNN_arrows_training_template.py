# -*- coding: utf-8 -*-
"""
Detection of arrows for robot control using CNN and ROS

This is template with example code to open and load the database.
You need to create your CNN model, train it, test it and save the model.

@author: Uriel Martinez-Hernandez
"""

# Load required packages
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma


from PIL import Image                                                            
import glob

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


np.random.seed(7)
    

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

# folder names of train and testing images
imageFolderTrainingPath = os.path.join('Database_arrows', 'train')
imageFolderTestingPath = os.path.join('Database_arrows', 'validation')
imageTrainingPath = []
imageTestingPath = []

# full path to training and testing images
for i in range(len(namesList)):
    trainingLoad = os.path.join(imageFolderTestingPath, namesList[i])
    testingLoad = os.path.join(imageFolderTestingPath, namesList[i])
    print(f"The training path is {trainingLoad} and the testingLoad is {testingLoad}")
    for trainImage in os.listdir(trainingLoad):
        imageTrainingPath.append(os.path.join(trainingLoad, trainImage))
    for testImage in os.listdir(testingLoad):
        imageTestingPath.append(os.path.join(testingLoad, testImage))
    
# print number of images for training and testing
print(f"There are {len(imageTrainingPath)} training images")
print(f"There are {len(imageTestingPath)} testing images")

# resize images to speed up training process
updateImageSize = [128, 128]
tempImg = Image.open(imageTrainingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.LANCZOS)
[imWidth, imHeight] = tempImg.size

# create space to load training images
x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth, 1))
# create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))


# load training images
for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.LANCZOS)
    x_train[i, :, :, 0] = np.array(tempImg, 'f')
    
# load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.LANCZOS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')


# create space to load training labels
y_train = np.zeros((len(x_train),))
# create space to load testing labels
y_test = np.zeros((len(x_test),))


# load training labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos = countPos + 1
    
# load testing labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
# convert training labels to one-hot format
y_train = tf.keras.utils.to_categorical(y_train, len(namesList))
# convert testing labels to one-hot format
y_test = tf.keras.utils.to_categorical(y_test, len(namesList))
        

# Creat your CNN model here composed of convolution, maxpooling, fully connected layers.
model = Sequential()

#layer 1
# 32 filters with a size of 3x3. The weight of the filters deternined by NN
#relu activation function - Add non lineratiy to network
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2,2)))

## Flatten infor and create my fully connected layer for finding out desired
#probabilites
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile, fit and evaluate your CNN model.
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
#batch_size - Is the number of training examples that is used to peform one step of
#stochastic gradient descent, Only use a subset of the training dateseet
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
score = model.evaluate(x_test, y_test, batch_size=5)

# Stores your model in a specific path
idFile = "ee30241_arrow_recognition.h5"
model.save(idFile)
# Display the accuracy achieved by your CNN model
print(f"The model metric name is {model.metrics_names[1]}\n The score is {score[1]*100}")

# Plot accuracy plots
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt

print('OK')
import argparse
import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gc; gc.collect()
print("pass")


# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'
base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training data/'
import time
import augmentation
# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training Data/'
base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'
base_path = '/data/'
base_path = '/Users/sumitasok/ml_data/Self-Driving-Car/Behavioural-Cloning/data/'
base_path = '/Users/sumitasok/ml_data/Self-Driving-Car/Behavioural-Cloning/data_orig/'
training_data_base_path = base_path
timestamp = str(int(time.time()*1000000))
print("file identifier: ", timestamp)
results_file = "/src/results/model-" + timestamp
results_file = "results/model-" + timestamp


def image_process(current_path):
    image = mpimg.imread(current_path)

    cropped = cv2.resize(image[60:140,:], (320, 80))
    
    R = cropped[:,:,0]
    G = cropped[:,:,1]
    B = cropped[:,:,2]
    thresh = (200, 255)
    rbinary = np.zeros_like(R)
    gbinary = np.zeros_like(G)
    rbinary[(R > thresh[0]) & (R <= thresh[1])] = 1
    
    return np.dstack((rbinary, gbinary, gbinary))

import augmentation

samples_per_epoch = 0

BATCH_SIZE = 128
EPOCHS = 4

# Set Validation_flag = [True|False]
def generator(samples, batch_size=32, validation_flag = False):
    print("generator called")
    num_samples = len(samples)
    # import pdb; pdb.set_trace()
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        global samples_per_epoch
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = base_path+ 'IMG/'+batch_sample[0].split('/')[-1]
                center_angle = float(batch_sample[3])
                images.append(image_process(name))
                angles.append(center_angle)
                
            agl_images = []
            agl_measurements = []
            str_imgs = images
            str_msr = angles

            if validation_flag == False:
                # Data Augmentation
                str_imgs, str_msr, agl_imgs, agl_msr = augmentation.split_straight_angle(images, angles)
                str_images, str_measurements = str_imgs, str_msr # augmentation.remove_excess_straigth_drive(str_imgs, str_msr, len(agl_imgs)/len(str_imgs))
                agl_images, agl_measurements = augmentation.invert_images_and_measurements(agl_imgs, agl_msr)

            X_train = np.array(str_imgs + agl_images)
            y_train = np.array(str_msr + agl_measurements)

            samples_per_epoch = len(X_train)
            # print("Shape: ", X_train[1].shape)
            yield shuffle(X_train, y_train)

print("pass")

import preprocessing as pro

samples = []
samples_per_epoch = 0
with open(training_data_base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        if (float(line[3]) > 0.34 or float(line[3]) < -0.34):
            samples_per_epoch += 2
        else:
            samples_per_epoch += 1

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# initialize generators
train_gen = generator(train_samples ,batch_size=BATCH_SIZE)
val_gen = generator(validation_samples ,batch_size=BATCH_SIZE, validation_flag = True)
# test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers.advanced_activations import ELU
model = Sequential()
# normaliastion of training data.
# https://keras.io/layers/convolutional/#cropping2d
# model.add(Cropping2D(cropping=((64, 23),(0, 0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(80, 320, 3)))
model.add(Conv2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(4,3,3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(127))
# model.add(Dropout(0.5))
model.add(Dense(84))
# model.add(ELU)
model.add(Dense(24))
# model.add(ELU)
# model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
print("pre fit_generator")
# model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)
print("samples_per_epoch: ", samples_per_epoch)
model.fit_generator(train_gen, samples_per_epoch = samples_per_epoch,
                    nb_epoch=EPOCHS, validation_data = val_gen, nb_val_samples = len(validation_samples))
#     history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040, 
#                                   nb_epoch=5, verbose=2, callbacks=[checkpoint])

# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
model_json = model.to_json()
with open(results_file +".json", "w") as json_file:
  json_file.write(model_json)
model.save(results_file +'.h5')
model.summary()
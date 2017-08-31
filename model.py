import argparse
import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import math

import gc; gc.collect()

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = base_path+ 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                image_resized = cv2.resize(center_image[60:140,:], (64,64))
                center_angle = float(batch_sample[3])
                images.append(image_resized)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'
base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training data/'
import time
import augmentation


# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training Data/'
base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'training_data_base_path',
    type=str,
    help='Path to training data, this dir should have /IMG and /driving_log.csv'
)

lines = []
training_data_base_path = base_path
with open(training_data_base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

samples = []
with open(training_data_base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


for line in train_samples:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    # image = cv2.resize(image[60:140,:], (64,64))
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Data Augmentation
str_imgs, str_msr, agl_imgs, agl_msr = augmentation.split_straight_angle(images, measurements)
str_images, str_measurements = str_imgs, str_msr # augmentation.remove_excess_straigth_drive(str_imgs, str_msr, len(agl_imgs)/len(str_imgs))
agl_images, agl_measurements = augmentation.invert_images_and_measurements(agl_imgs, agl_msr)

# X_train = np.array(images)
# y_train = np.array(measurements)
X_train = np.array(str_images + agl_images)
y_train = np.array(str_measurements + agl_measurements)

print("straight images count: ", len(str_images), "measurements count: ", len(str_measurements), len(X_train))
print("angle images count: ", len(agl_images), "measurements count: ", len(agl_measurements), len(y_train))

images = []
measurements = []
for line in validation_samples:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    # image = cv2.resize(image[60:140,:], (64,64))
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_val = np.array(images)
y_val = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers.advanced_activations import ELU

model = Sequential()
# normaliastion of training data.
# https://keras.io/layers/convolutional/#cropping2d
model.add(Cropping2D(cropping=((64, 23),(0, 0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(100, 180, 3)))
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

# model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)
model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(X_train)), nb_epoch=1, validation_data = valid_generator, nb_val_samples = len(X_val))

timestamp = str(time.time()*1000000)
print("file identifier: ", timestamp)
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
model_json = model.to_json()
with open("results/model-"+ timestamp +".json", "w") as json_file:
  json_file.write(model_json)
model.save('results/model-'+ timestamp +'.h5')
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='results/model-'+timestamp+'.png')

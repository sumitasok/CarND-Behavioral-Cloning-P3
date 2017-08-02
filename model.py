import argparse
import numpy as np
import csv
import cv2
import time
import augmentation


# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training Data/'
base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'training_data_base_path',
    type=str,
    help='Path to training data, this dir should have /IMG and driving_log.csv'
)

lines = []
training_data_base_path = base_path
with open(training_data_base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + 'IMG/' + filename
    # current_path = base_path + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Data Augmentation
str_imgs, str_msr, agl_imgs, agl_msr = augmentation.split_straight_angle(images, measurements)
str_images, str_measurements = augmentation.remove_excess_straigth_drive(str_imgs, str_msr, len(agl_imgs)/len(str_imgs))
agl_images, agl_measurements = augmentation.invert_images_and_measurements(agl_imgs, agl_msr)

# X_train = np.array(images)
# y_train = np.array(measurements)
X_train = np.array(str_images + agl_images)
y_train = np.array(str_measurements + agl_measurements)

print("images count")
print(len(str_images), len(str_measurements), len(X_train))
print("measurements count")
print(len(agl_images), len(agl_measurements), len(y_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# normaliastion of training data.
# https://keras.io/layers/convolutional/#cropping2d
model.add(Cropping2D(cropping=((64, 23),(0, 0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(100, 180, 3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(127))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

timestamp = str(time.time()*1000000)
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
model_json = model.to_json()
with open("model-h5/model-"+ timestamp +".json", "w") as json_file:
  json_file.write(model_json)
model.save('model-h5/model-'+ timestamp +'.h5')
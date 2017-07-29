import argparse
import numpy as np
import csv
import cv2

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

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    image_resized = cv2.resize(image[60:140,:], (64,64))
    images.append(image_resized)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras.regularizers as regularizers

model = Sequential()
# normalisation of training data.
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(64, 64, 3)))
model.add(Convolution2D(3,3,3, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(3,3,3, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(3,3,3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(127))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')
import argparse
import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'
# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training data/'

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


# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = base_path + 'IMG/' + filename
#     image = cv2.imread(current_path)
#     image_resized = cv2.resize(image[60:140,:], (64,64))
#     images.append(image_resized)
#     measurement = float(line[3])
#     measurements.append(measurement)

# X_train = np.array(images)
# y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras.regularizers as regularizers

train_generator = generator(train_samples, 32)
validation_generator = generator(validation_samples, 32)

model = Sequential()
# normalisation of training data.
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(64, 64, 3), output_shape=(64, 64, 3)))
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
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
import argparse
import numpy as np
import csv
import cv2
import time
import augmentation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import preprocessing as pp
from sklearn.utils import shuffle

import gc; gc.collect()

# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training Data/'
# base_path =  '/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/data/'
base_path = '/Users/sumitasok/ml_data/Self-Driving-Car/Behavioural-Cloning/data/'
base_path = '/Users/sumitasok/ml_data/Self-Driving-Car/Behavioural-Cloning/data_orig/'
base_path = '/input/data_orig/'
base_path = '/input/data/'
# base_path = '/input/'

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'training_data_base_path',
    type=str,
    help='Path to training data, this dir should have /IMG and driving_log.csv'
)

EPOCHS = 4

import augmentation

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
                center_angle = float(batch_sample[3])
                images.append(pp.AutoCannyGaussianBlurSobelYRGB(name))
                angles.append(center_angle)
                
            # Data Augmentation
            str_imgs, str_msr, agl_imgs, agl_msr = augmentation.split_straight_angle(images, angles)
            str_images, str_measurements = str_imgs, str_msr # augmentation.remove_excess_straigth_drive(str_imgs, str_msr, len(agl_imgs)/len(str_imgs))
            agl_images, agl_measurements = augmentation.invert_images_and_measurements(agl_imgs, agl_msr)

            X_train = np.array(str_imgs + agl_images)
            y_train = np.array(str_msr + agl_measurements)
            for i in range(0, len(X_train)):
                yield (X_train[i], y_train[i])


lines = []
training_data_base_path = base_path
for i in list(range(7))
    with open(training_data_base_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# counter = len(lines)
counter = 100
restricted = False

images = []
measurements = []

# import progressbar

# https://stackoverflow.com/questions/3160699/python-progress-bar
# https://pypi.python.org/pypi/progressbar2
# bar = progressbar.ProgressBar()
from tqdm import tqdm
for line in tqdm(lines):
    if counter == 0:
        break
    if restricted == True:
        counter -= 1
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + 'IMG/' + filename
    # current_path = base_path + filename
    image = cv2.imread(current_path)
    image = np.asarray(image)
    # image = pp.AutoCannyGaussianBlurSobelYRGB(image)
    # image = pp.SobelYRGB(image)
    image = pp.CropSky(image)


    # plt.imshow(image)
    # plt.savefig('results/videos/' + filename + '.png')
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

X_train, y_train = shuffle(X_train, y_train)

print("straight images count: ", len(str_images), "measurements count: ", len(str_measurements), len(X_train))
print("angle images count: ", len(agl_images), "measurements count: ", len(agl_measurements), len(y_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
# from keras.layers.advanced_activations import ELU

# model = Sequential()
# # normaliastion of training data.
# # https://keras.io/layers/convolutional/#cropping2d
# # model.add(Cropping2D(cropping=((64, 23),(0, 0)), input_shape=(80, 320, 3)))
# # model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(160, 320, 3)))
# model.add(Conv2D(6,5,5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(6,5,5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(4,3,3, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(127))
# model.add(Dropout(0.5))
# model.add(Dense(84))
# model.add(ELU())
# model.add(Dense(24))
# model.add(ELU())
# model.add(Dropout(0.5))
# model.add(Dense(1))



model = Sequential()
# normaliastion of training data.
# https://keras.io/layers/convolutional/#cropping2d
# model.add(Cropping2D(cropping=((64, 23),(0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(80, 320, 3)))
model.add(Conv2D(16,8,8, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(32,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(64,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(127))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)

timestamp = str(time.time()*1000000)
print("file identifier: ", timestamp)
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
model_json = model.to_json()
with open("/output/model-"+ timestamp +".json", "w") as json_file:
  json_file.write(model_json)
model.save('/output/model-'+ timestamp +'.h5')
model.summary()
# from keras.utils import plot_model
# plot_model(model, to_file='results/model-'+timestamp+'.png')
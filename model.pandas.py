import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import gc

colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('/Users/sumitasok/Documents/Self-Driving Car/Behavioural Cloning/Training data/driving_log.csv', skiprows=[0], names=colnames)
center_images = data.center.tolist()
steering_angles = data.steering.tolist()

center_shuffled, steering_shuffled = shuffle(center_images, steering_angles)
center, X_valid, steering, y_valid = train_test_split(center_shuffled, steering_shuffled, test_size = 0.10, random_state = 100) 
gc.collect()

def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(center, steering)
        for i in range(batch_size):
          choice = int(np.random.choice(len(data),1))
          print(data[choice].strip())
          batch_train[i] = crop_resize(mpimg.imread(data[choice].strip()))
          batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))

        yield batch_train, batch_angle

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
      data, angle = shuffle(data,angle)
      for i in range(batch_size):
        rand = int(np.random.choice(len(data),1))
        batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
        batch_angle[i] = angle[rand]
      yield batch_train, batch_angle

def crop_resize(image):
	cropped = cv2.resize(image[60:140,:], (64,64))
	return cropped

def main(_):
	data_generator = generator_data(128)
	valid_generator = generator_valid(X_valid, y_valid, 128)


	model = Sequential()
	# normalisation of training data.
	# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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
	# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

	# model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)
	model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(center)), nb_epoch=1, validation_data = valid_generator, nb_val_samples = len(X_valid))

	# model_json = model.to_json()
	# with open("model.json", "w") as json_file:
	#     json_file.write(model_json)
	model.save("model.h5")
	gc.collect()

if __name__ == '__main__':
  tf.app.run()
import copy
import cv2
from sklearn.model_selection import train_test_split

def invert_images_and_measurements(images, measurements):
	images = images + list(map(flip_cv2_image_veritically, images))
	measurements = measurements + list(map(flip_measurements, measurements))
	return images, measurements

def flip_cv2_image_veritically(image):
	image = cv2.flip(image, 1)
	return image

def flip_measurements(measurement):
	return measurement * -1

def remove_excess_straigth_drive(images, measurements, perc):
	drive_omit, drive_choosen, measurements_omit, measurements_choosen = train_test_split(images, measurements, test_size = perc, random_state = 100) 

	return drive_choosen, measurements_choosen

def split_straight_angle(images, measurements):
	str_drive = []
	str_measurements = []

	angle_drive = []
	angle_measurements = []

	for image, measurement in zip(images, measurements):
		if (measurement > 0.34 or measurement < -0.34):
			angle_drive.append(image)
			angle_measurements.append(measurement)
		else:
			str_drive.append(image)
			str_measurements.append(measurement)

	return str_drive, str_measurements, angle_drive, angle_measurements
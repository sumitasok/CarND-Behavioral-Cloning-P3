from PIL import Image

def hello():
	print('hello')

def crop_like_keras_crop2D(input_filename, output_filename, top_crop, bottom_crop, left_crop, right_crop):
	img = Image.open(image_file_name)
	x_length, y_length = img.size
	img.crop((left_crop, top_crop, x_length - right_crop, y_length - bottom_crop))
	img.save(output_filename)
	img.close()
from PIL import Image

def hello():
	print('hello')

# crop the image using the margin format that keras.cropping2D uses.
# makes it simpler to port the cropping configurations.
# https://keras.io/layers/convolutional/#cropping2d
# http://matthiaseisen.com/pp/patterns/p0202/
def crop_like_keras_crop2D(input_filename, output_filename, top_crop, bottom_crop, left_crop, right_crop):
	img = Image.open(input_filename)
	x_length, y_length = img.size
	cropped_image = img.crop((left_crop, top_crop, x_length - right_crop, y_length - bottom_crop))
	cropped_image.save(output_filename)
	img.close()
	return output_filename
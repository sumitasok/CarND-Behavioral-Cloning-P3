from PIL import Image
# from cv2 import getPerspectiveTransform, warpPerspective
import cv2

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

#   src = np.float32([
#        [850, 320],
#        [865, 450],
#        [533, 350],
#        [535, 210]
#    ])
#   src = np.float32([
#        [870, 240],
#        [870, 370],
#        [520, 370],
#        [520, 240]
#    ])
def warp(img, src_points, dst_points, img_size=None):

    if img_size == None:
        img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
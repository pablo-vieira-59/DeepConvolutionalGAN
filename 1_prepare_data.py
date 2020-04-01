import numpy as np
import cv2

def prepare_images(n_images:int, output_shape:tuple,images_path:str, image_extension:str, output_path:str, output_name:str):
    images = []
    for i in range(1, n_images+1):
        image = cv2.imread(images_path+str(i)+image_extension)
        image = cv2.resize(image, output_shape)
        image = (image - 127.5) + 127.5
        images.append(image)
    np.save(output_path+output_name, images)
    return images

# Paths
images_path = 'train_images/128/'
output_path = 'train_images/'

# Misc Variables
n_images = 100
image_extension = '.png'
output_shape = (256,128)
output_name = 'images_128.npy'

prepare_images(n_images, output_shape, images_path, image_extension, output_path, output_name)
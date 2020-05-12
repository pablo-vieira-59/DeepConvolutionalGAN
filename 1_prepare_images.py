import glob
import cv2 as cv
import numpy as np

def show_images(images :np.array):
    print('Showing Images ...')
    for image in images:
        cv.imshow('Window', image)
        cv.waitKey(15)

def read_images(images_path: str, extension :str):
    print('Reading Images ...')
    path = glob.glob(images_path + extension)
    images = []
    for img in path:
        n = cv.imread(img)
        images.append(n)            

    return np.array(images)

def resize_images(images :np.array, resize_shape :tuple):
    print('Resizing Images ...')
    n_images = []
    for image in images:
        n = cv.resize(image, resize_shape)
        n_images.append(n)
    return np.array(n_images)

def flip_images(images :np.array):
    print('Flipping Images ...')
    n_images = []
    for image in images:
        n_flip = cv.flip(image, 1)
        n_images.append(n_flip)
        n_images.append(image)
    return np.array(n_images)

def normalize_images(images :np.array):
    print('Normalizing Images ...')
    n_images = []
    for image in images:
        n = (image - 127.5) / 127.5
        n_images.append(n)

    return np.array(n_images)

def apply_gray_scale(images :np.array):
    print('Gray Scaling Images ...')
    n_images = []
    for image in images:
        n = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        n_images.append(n)
    return np.array(n_images)

def save_preview(images :np.array, save_path:str):
    print('Saving Preview ...')
    for i in range(0, len(images)):
        cv.imwrite(save_path + str(i) + '.png', images[i])

def save_images(images :np.array, save_path :str):
    print('Saving Npy ...')
    np.save(save_path, images)

# Paths
extension = "/*.png"
images_path = 'images/data'
save_path = 'images_npy/images_w.npy'
preview_path = 'preview/'

# Resize Shape
r_shape = (16, 32)

# Prepare Images
images = read_images(images_path, extension)
images = resize_images(images, r_shape)
images = flip_images(images)


save_preview(images ,preview_path)
#show_images(images)

images = normalize_images(images)

save_images(images, save_path)

print('Dataset Shape :', images.shape)


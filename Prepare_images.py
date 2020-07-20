import glob
import cv2 as cv
import numpy as np

def show_images(images :np.array):
    print('Showing Images ...')
    for image in images:
        cv.imshow('Window', image)
        cv.waitKey(150)

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
        #n_images.append(image)
    return np.array(n_images)

def normalize_images(images :np.array):
    print('Normalizing Images ...')
    images = (images - 127.5) / 127.5
    return images

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

def crop_images(images :np.array, crop_size :int):
    print('Cropping Images ...')
    n_images = []
    for image in images:
        y,x,_ = image.shape
        startx = x//2-(crop_size//2)
        starty = y//2-(crop_size//4)    
        n = image[starty:starty+crop_size, startx:startx+crop_size]
        n_images.append(n)
    return np.array(n_images)

def apply_noise(images :np.array, noise_factor:float):
    noise = np.random.uniform(size=images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3])
    noise = np.reshape(noise, images.shape) * noise_factor
    images = np.add(images, noise)
    return images

def restore_images(images :np.array):
    images_res = []
    for image in images:
        n = image
        n = image[:,:,:] * 127.5 + 127.5
        n = np.array(n, dtype='uint8')
        images_res.append(n)
    images_res = np.array(images_res)
    return images_res


# Paths
extension = "/*.png"
images_path = 'images/cars'
save_path = 'images_npy/cars.npy'
preview_path = 'preview/'

# Resize Shape
r_shape = (64, 128)

# Prepare Images
images = read_images(images_path, extension)
images = crop_images(images, 256)
images = resize_images(images, r_shape)
#images = apply_gray_scale(images)
save_images(images, save_path)
#save_preview(images, preview_path)

print('Dataset Shape :', images.shape)


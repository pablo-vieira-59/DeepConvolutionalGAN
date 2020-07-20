import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras import backend as K

def make_noise(n_samples :int, n_dim :int):
    noise = np.random.randn(n_samples * n_dim)
    noise = np.reshape(noise, [n_samples, n_dim])
    return noise

def restore_images(images :list, preview_shape: tuple):
    images_res = []
    for image in images:
        n = image
        n = image[:,:,:] * 127.5 + 127.5
        n = np.array(n, dtype='uint8')
        n = cv.resize(n, preview_shape, interpolation=cv.INTER_AREA)
        images_res.append(n)
    images_res = np.array(images_res)
    return images_res

def show_images(images :np.array, window, preview_shape):
    images = restore_images(images, preview_shape)
    images = np.split(images, 2)
    line_1 = np.concatenate(images[0], axis=1)
    line_2 = np.concatenate(images[1], axis=1)
    frame = np.concatenate([line_1, line_2], axis=0)
    cv.imshow(window, frame)
    cv.waitKey(15)

def create_fake_images(noise :np.array, gen_model):
    pred = gen_model.predict(noise)
    return pred

def sample_data(n_samples: int, dataset :np.array):
    n_batches = len(dataset)//n_samples
    X = np.split(dataset, n_batches)
    return X

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def full_dataset(data, n_dim, gen_model):
    noise = make_noise(len(data), n_dim)
    fake_images = create_fake_images(noise, gen_model)
    X = np.append(data, fake_images, axis=0)
    Y = np.append(np.ones(len(data)), np.zeros(len(fake_images)), axis=0)
    return X, Y

def normalize_images(images :np.array):
    n = (images - 127.5) / 127.5
    return n.astype("float32")

def sigmoid_noise(n_samples, n_dim):
    noise = make_noise(n_samples, n_dim)
    noise = 1/(1 + np.exp(-noise))
    return noise

def apply_noise(images :np.array, noise_factor:float):
    noise = np.random.uniform(size=images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3])
    noise = np.reshape(noise, images.shape) * noise_factor
    images = np.add(images, noise)
    return images

def save_preview(path :str, images :np.array, preview_shape :tuple, epoch):
    images = restore_images(images, preview_shape)
    images = np.split(images, 2)
    line_1 = np.concatenate(images[0], axis=1)
    line_2 = np.concatenate(images[1], axis=1)
    frame = np.concatenate([line_1, line_2], axis=0)
    cv.imwrite(path + str(epoch) + '.png', frame)

# Define Seeds
tf.random.set_seed(59)
np.random.seed(59)
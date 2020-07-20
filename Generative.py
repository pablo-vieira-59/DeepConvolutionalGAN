import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import cv2 as cv
import numpy as np
import tensorflow as tf

import Utils as gu
import Models as models

from tensorflow.keras.metrics import BinaryAccuracy, Accuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential , load_model

# Seeds
tf.random.set_seed(59)
np.random.seed(59)

# Vars
multiplier = 3
latent_dim = 128
n_epochs = 500
n_batch = 10
n_preview = 10
fixed_latent = gu.make_noise(n_preview, latent_dim)
threshold = 1

# Paths
dataset_path = "images_npy/cars/cars_32.npy"
gen_save_path = "saved_models/cars_dec.h5"
pre_save_path = "saved_models/cars_16.h5"

# Dataset
#-MNIST
#(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#dataset = np.concatenate([x_train, x_test])
#dataset = np.reshape(dataset, (-1, 28, 28, 1))

#-Custom
dataset = np.load(dataset_path)

#-Normalize Dataset - (-1,+1)
dataset = gu.normalize_images(dataset)

# Shapes
print('Dataset Shape :', dataset.shape)
initial_shape = [4,8,128]
img_shape = dataset[0].shape
channels = img_shape[2]
preview_shape = (256,128)

# Optimizers
gen_opt = Adam(learning_rate=0.00005, beta_1=0.4)
dis_opt = Adam(learning_rate=0.00005, beta_1=0.4)

# Discriminator
dis_model = models.discriminator_model(img_shape, 64, 3, multiplier, False)
dis_model.compile(optimizer=dis_opt, loss='binary_crossentropy', metrics=['accuracy'])

# Generator
gen_model = models.generator_model(latent_dim, initial_shape, 256, 3, multiplier, channels, False)

# Adversarial
gan_model = models.adversarial_model(gen_model, dis_model)
gan_model.compile(optimizer=gen_opt, loss='binary_crossentropy', metrics=['accuracy'])

# Preview Window
window_name = 'Preview'
cv.namedWindow(window_name)
cv.moveWindow(window_name, 0, 0)

def draw_preview():
    generated_images = gen_model(fixed_latent)
    gu.show_images(generated_images, window_name, preview_shape)
        
def train_step(X, i):
    d = []
    g = []
    for j in range(0, len(X)):
        X_real = X[j]
        #if i < n_epochs:
        #    X_real = gu.apply_noise(X[j], 0.5 * (1 - (i/n_epochs)))

        Y_real = np.zeros([n_batch,1])
        #Y_real += 0.05 * np.reshape(np.random.randn(n_batch), [n_batch,1])

        Y_fake = np.ones([n_batch,1])
        X_fake = gu.create_fake_images(gu.make_noise(n_batch, latent_dim), gen_model)

        r_loss = dis_model.evaluate(X_real[0:n_batch], Y_real[0:n_batch], verbose=0)[0]
        f_loss = dis_model.evaluate(X_fake[0:n_batch], Y_fake[0:n_batch], verbose=0)[0]

        gen_x = gu.make_noise(n_batch, latent_dim)
        g_loss = gan_model.evaluate(gen_x, Y_real[0:n_batch], verbose=0)[0]

        dis_model.train_on_batch(X_real, Y_real)[0]
        dis_model.train_on_batch(X_fake, Y_fake)[0]

        noise = gu.make_noise(n_batch, latent_dim)
        valids = np.zeros([n_batch, 1])
        gan_model.train_on_batch(noise, valids)[0]
            
        draw_preview()

        print('Epoch:%i/%i Batch:%i/%i r_loss:%.3f f_loss:%.3f g_loss:%.3f' % (n_epochs, i, len(X), j, r_loss, f_loss, g_loss))

def train():
    X = gu.sample_data(n_batch, dataset)
    for i in range(0, n_epochs):
        train_step(X, i)

        if i % 10 == 0 or i == n_epochs-1:
            gen_model.save(gen_save_path)
            images = gu.create_fake_images(fixed_latent, gen_model)
            gu.save_preview('preview/', images, preview_shape, i)

# Start Training
train()
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import cv2 as cv
import numpy as np
import tensorflow as tf
import keras.backend as k

from sklearn.metrics import accuracy_score
from tensorflow_core.keras.optimizers import Adam
from tensorflow_core.keras.models import Sequential , load_model
from tensorflow_core.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense , Dropout, Flatten, LeakyReLU , Reshape
 
def generator_model(n_dim, in_shape, multiplier):
    model = Sequential()
    model.add(Dense(in_shape[0]*in_shape[1]*in_shape[2] , input_shape=(n_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 2048x1

    model.add(Reshape(in_shape))
    # 8x4x64

    model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 16x8x256

    for i in range(0, multiplier):
        model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
    
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='tanh', use_bias=False))
    model.summary()
    return model

def discriminator_model(optimizer, multiplier, img_shape):
    model = Sequential()

    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # 128x64x16

    for i in range(0, multiplier):
        model.add(Conv2D(filters=128, kernel_size=(5,5), strides=2, padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        # 64x32x32
        
    model.add(Flatten())
    model.add(Dense(1))

    #model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def adversarial_model(gen_model, dis_model, optimizer):
    dis_model.trainable = False

    model = Sequential()
    model.add(gen_model)
    model.add(dis_model)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def make_noise(n_samples :int, n_dim :int):
    noise = np.random.randn(n_samples * 100)
    noise = noise.reshape(n_samples, 100)
    return noise

def restore_images(images :list):
    images_res = []
    for image in images:
        n = image
        n = image[:,:,:] * 127.5 + 127.5
        n = np.array(n, dtype='uint8')
        n = cv.resize(n, preview_shape)
        images_res.append(n)
    images_res = np.array(images_res)
    return images_res
    
def show_images(images :np.array, window):
    images = restore_images(images)
    images = np.split(images, 2)
    line_1 = np.concatenate(images[0], axis=1)
    line_2 = np.concatenate(images[1], axis=1)
    frame = np.concatenate([line_1, line_2], axis=0)
    cv.imshow(window, frame)
    cv.waitKey(15)

def create_fake_images(noise :np.array, model):
    pred = model.predict(noise)
    return pred

def sample_data(n_samples: int, dataset :np.array):
    n_batches = len(dataset)//n_samples
    X = np.split(dataset, n_batches)
    return X ,n_batches

def calc_metrics(dis_model, gen_model, X, n_dim):
    Y = np.ones(len(X))
    pred = dis_model.predict_classes(X)
    r = accuracy_score(Y, pred)

    noise = make_noise(len(X), n_dim)
    X_fake = create_fake_images(noise, gen_model)
    Y_fake = np.ones(len(X))
    pred = dis_model.predict_classes(X_fake)
    g = accuracy_score(Y_fake, pred)
    
    return r, g

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, n_batch, n_dim, discriminator, generator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([n_batch, n_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss, gen_loss

def train_tf(n_batch, n_epochs, gen_model, dis_model, dataset :np.array, window :str, n_dim :int, n_preview :int, gen_opt, dis_opt):
    X , n_batches = sample_data(n_batch, dataset)
    fixed_seed = make_noise(n_batch, n_dim)
    for i in range(0, n_epochs):
        disc_loss = []
        gen_loss = []
        for j in range(0, n_batches):
            d_l, g_l = train_step(X[j], n_batch, n_dim, dis_model, gen_model, gen_opt, dis_opt)
            disc_loss.append(d_l)
            gen_loss.append(g_l)
            images = create_fake_images(fixed_seed, gen_model)
            show_images(images, window)
        
        d_a, g_a = calc_metrics(dis_model, gen_model, dataset, n_dim)
        d = sum(disc_loss)/len(disc_loss)
        g = sum(gen_loss)/len(gen_loss)
        print('Epoch :%d/%d, Disc Loss:%.3f, Gen Loss:%.3f, Disc Acc:%.3f, Gen Acc:%.3f' % (n_epochs, i, d, g, d_a, g_a))

        if i%10 == 0 and i != 0:
            gen_model.save(save_path)

# Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optmizers
gen_opt = Adam(learning_rate=0.0001 ,beta_1=0.4)
dis_opt = Adam(learning_rate=0.00001, beta_1=0.4)

# Preview Window
window = 'Preview'
cv.namedWindow(window)
cv.moveWindow(window, 500, 200)
n_preview = 8

# Training Variables
n_batch = 10
n_epochs = 100000
n_dim = 100

# Paths
dataset_path = "images_npy/images_w.npy"
save_path = "saved_models/model_w.h5"

# Dataset
data = np.load(dataset_path)
np.random.shuffle(data)

# Shapes
preview_shape = (128,256)
multiplier = 1
in_shape = [8,4,256]
img_shape = data[0].shape
img_h = img_shape[0]
img_w = img_shape[1]
img_c = 1
try:
    img_c = img_shape[2]
except:
    data = np.reshape(data, (len(data), img_h, img_w, 1))
img_shape = [img_h,img_w,img_c]
print('Shape :', data.shape)

# Models
#gen_model = generator_model(n_dim, in_shape, multiplier)
gen_model = load_model(save_path)
dis_model = discriminator_model(dis_opt, multiplier, img_shape)

# Training
train_tf(n_batch, n_epochs, gen_model, dis_model, data, window, n_dim, n_preview, gen_opt, dis_opt)
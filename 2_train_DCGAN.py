import os, time , cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from numpy.random import randn, randint
from numpy import expand_dims, zeros, ones

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Flatten, LeakyReLU, Reshape, Dropout


def make_generator_upsample():
	model = Sequential()
	gen_shape = [8, 4, 256]
	
	model.add(Dense(gen_shape[0] * gen_shape[1] * gen_shape[2],use_bias=False, input_shape=[dim_noise, ]))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape(target_shape=gen_shape))
	# 8x4x128

	model.add(UpSampling2D(size=2))
	model.add(Conv2D(filters=256, kernel_size=4,padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 16x8x128

	model.add(UpSampling2D(size=2))
	model.add(Conv2D(filters=128, kernel_size=4,padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 32x16x128

	'''
	model.add(UpSampling2D(size=2))
	model.add(Conv2D(filters=64, kernel_size=4,padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.5))
	# 64x32x128
	'''
	model.add(Conv2DTranspose(filters=3, kernel_size=4, strides=1,padding='same', activation='tanh', use_bias=False))

	return model


def make_generator():
	model = Sequential()
	gen_shape = [4, 2, 256]
	model.add(Dense(gen_shape[0] * gen_shape[1] * gen_shape[2],use_bias=False, input_shape=[dim_noise, ]))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape(target_shape=gen_shape))
	# 4x2x256

	model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 8x4x256

	model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 16x8x128

	model.add(Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 32x16x128

	model.add(Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 64x32x128

	model.add(Conv2DTranspose(filters=32, kernel_size=8, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 128x64x128

	model.add(Conv2DTranspose(filters=16, kernel_size=8, strides=2, padding='same', use_bias=False))
	model.add(BatchNormalization(momentum=0.7))
	model.add(LeakyReLU(alpha=0.2))
	# 256x128x16

	model.add(Conv2DTranspose(filters=img_c, kernel_size=8, strides=1, padding='same', activation='tanh', use_bias=False))
	return model


def make_discriminator():
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=4, strides=2, padding='same', input_shape=img_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.2))
	# 128x64x8

	model.add(Conv2D(filters=64, kernel_size=4, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.2))
	# 64x32x16

	model.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.2))
	# 32x16x32
	
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy', metrics=['accuracy'])
	return model


def make_gan():
	d_model.trainable = False
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	model.compile(optimizer=generator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	return model


def make_noise(n_samples: int):
    noise = randn(dim_noise * n_samples)
    noise = noise.reshape(n_samples, dim_noise)
    return noise


def restore_images(images):
    for i in range(0, len(images)):
        image = images[i]
        image = image[:, :, :] * 127.5 + 127.5
        image = np.array(image, dtype='uint8')
    return images


def make_preview(images):
    images = restore_images(images)
    images = np.split(images, 2)
    line_1 = np.concatenate(images[0], axis=1)
    line_2 = np.concatenate(images[1], axis=1)
    frame = np.concatenate([line_1, line_2], axis=0)
    cv2.imshow(window, frame)
    cv2.waitKey(5)


def make_samples():
    n_batches = len(dataset)//n_batch
    Y = ones(len(dataset))
    X = np.split(dataset, n_batches)
    Y = np.split(Y, n_batches)
    return X, Y, n_batches


def make_samples_fake(n_batches):
	noise = make_noise(n_samples=len(dataset))
	#noise = fixed_noise_train
	X = g_model.predict(noise)
	Y = zeros(len(X))
	X = np.split(X, n_batches)
	Y = np.split(Y, n_batches)
	return X, Y


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)


def train():
	start = time.time()
	X, Y, n_batches = make_samples()
	for i in range(n_epoch):
		r_losses = []
		f_losses = []
		g_losses = []
		X_fake, Y_fake = make_samples_fake(n_batch)
		for j in range(n_batches):
			# Get Batches
			X_batch = X[j]
			Y_batch = Y[j]
			X_f_batch = X_fake[j]
			Y_f_batch = Y_fake[j]
			X_gan = make_noise(n_batch)
			Y_gan = ones(n_batch)

			# Train Discriminator
			d_loss_1, _ = d_model.train_on_batch(X_batch, Y_batch)
			d_loss_2, _ = d_model.train_on_batch(X_f_batch, Y_f_batch)

			# Train Generator
			g_loss, _ = gan_model.train_on_batch(X_gan, Y_gan)

			# Save in array to calculate Mean Loss
			r_losses.append(d_loss_1)
			f_losses.append(d_loss_2)
			g_losses.append(g_loss)

			# Preview Batch
			make_preview(X_f_batch)

		# Calculate Mean Loss
		r_loss = np.mean(r_losses)
		f_loss = np.mean(f_losses)
		g_loss = np.mean(g_losses)

		# Print Info
		now = time.time() - start
		print('Epoch:%d/%d, real_loss:%.3f, fake_loss:%.3f, gen_loss:%.3f, time:%.3f' %  (i+1, n_epoch, r_loss, f_loss, g_loss, now))

		if i % 100 == 0 and i != 0:
			g_model.save(model_path)


def train_step_tf(images):
	noise = make_noise(n_batch)
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = g_model(noise, training=True)

		real_output = d_model(images, training=True)
		fake_output = d_model(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
	make_preview(generated_images)
	return gen_loss.numpy(), disc_loss.numpy()


def train_tf():
	start = time.time()
	X, Y, n_batches = make_samples()
	for i in range(n_epoch):
		d_losses = []
		g_losses = []
		for j in range(n_batches):
			g_loss, d_loss = train_step_tf(X[j])
			d_losses.append(d_loss)
			g_losses.append(g_loss)
		g_loss = np.mean(g_losses)
		d_loss = np.mean(d_losses)
		print('Epoch: %d/%d, D_loss:%.3f, G_loss:%.3f' % (i, n_epoch, d_loss, g_loss))

		if i % 100 == 0 and i != 0:
			g_model.save(model_path)



# Misc Variables
cross_entropy = BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)

# Preview Variables
window = 'Preview'
cv2.namedWindow(window)
cv2.moveWindow(window, 500, 200)
n_preview = 32

# Image Shape
img_h = 256
img_w = 128
img_c = 3
img_shape = [img_h, img_w, img_c]

# Paths
dataset_path = 'train_images/images_128.npy'
model_path = 'models/gmodel.h5'

# Dataset
dataset = np.load(dataset_path)

# Training Variables
n_batch = 10
n_epoch = 100000

# Noise Variables
dim_noise = 128
fixed_noise = make_noise(n_preview)

# Models
g_model = make_generator()
#g_model = tf.keras.models.load_model(model_path, compile=False)
g_model._name = 'gen'
d_model = make_discriminator()
gan_model = make_gan()

train()

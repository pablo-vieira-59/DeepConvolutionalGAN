from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, BatchNormalization, Reshape, InputLayer, GlobalMaxPool2D,UpSampling2D, Flatten, MaxPool2D

def discriminator_model(input_shape, n_filters, kernel_size, multiplier, summary):
    filters = n_filters
    model = Sequential(name='Discriminator')
    model.add(InputLayer(input_shape=input_shape))

    for i in range(0, multiplier):
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        filters = filters*2

    model.add(GlobalMaxPool2D())
    model.add(Dense(1, activation='sigmoid'))
    if(summary):
        model.summary()
    return model

def generator_model(latent_dim, input_shape, n_filters, kernel_size, multiplier, channels, summary):
    filters = n_filters
    model = Sequential(name='Generator')
    model.add(InputLayer(input_shape=latent_dim))

    model.add(Dense(input_shape[0] * input_shape[1] * input_shape[2], use_bias=False))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape(input_shape))
    
    for i in range(0, multiplier):
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        filters = filters//2

    model.add(Conv2D(filters=channels, kernel_size=kernel_size, strides=1, padding='same', activation='tanh'))
    if(summary):
        model.summary()
    return model

def adversarial_model(generator, discriminator):
    discriminator.trainable = False
    model = Sequential(name='Adversarial')
    model.add(generator)
    model.add(discriminator)
    return model

def encoder_model(input_shape, multiplier, n_filters, kernel_size, latent_dim, summary):
    filters = n_filters
    model = Sequential(name='Encoder')
    model.add(InputLayer(input_shape=input_shape))

    for i in range(0, multiplier):
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu'))
        model.add(LeakyReLU(alpha=0.2))
        filters = filters * 2

    model.add(Flatten())
    model.add(Dense(latent_dim, activation='sigmoid'))

    if summary:
        model.summary()

    return model

def joined_model(model_a, model_b):
    model = Sequential(name='Joined_Model')
    model.add(model_a)
    model.add(model_b)
    return model

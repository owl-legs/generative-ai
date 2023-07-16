import glob
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

print('''loading training data''')

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

print(f'''training data dimmensions: {train_images.shape}''')
print(f'''number of images: {train_images.shape[0]}''')
print(f'''image dimmensions: {(train_images.shape[1], train_images.shape[2])}''')
print('''number of image channels: 1''')

print('''processing training data''')

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256

print('''creating training set from tensor slices''')

train_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE)

def make_generator():

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

print('''creating generator model''')

generator = make_generator()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

print('''displaying generated image''')

plt.imshow(generated_image[0, :, :, 0])
plt.show()

def make_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

print('''creating discriminator''')

discriminator = make_discriminator()
decision = discriminator(generated_image)
print(decision)



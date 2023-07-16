import glob
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

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

train_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_directory = './checkpoint_logs'
checkpoint_prefix = os.path.join(checkpoint_directory, "checkpoint")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dimmension = 100
number_of_examples_to_generate = 16

seed = tf.random.normal([number_of_examples_to_generate, noise_dimmension])
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dimmension])

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_grads = generator_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grads = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'''time for epoch {epoch + 1} is {time.time() - start}''')

    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)


def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig(f'''output/image_at_epoch_{epoch}.png''')
    plt.show()

train(train_data, EPOCHS)

#make a gif
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

def display_image(epoch):
    return PIL.Image.open(f'''output/image_at_epoch_{epoch}.png''')

display_image(EPOCHS)

gif_path = 'numbers.gif'

with imageio.get_writer(gif_path, mode="I") as writer:
    filenames = glob.glob("image*.png")
    filenames = sorted(filenames)

    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
import os
import time
import config
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from image_parser import ImageParser
from discriminator import Discriminator
from generator import Generator

from IPython import display

BATCH_SIZE = config.BATCH_SIZE

print("( ^..^)ﾉ opening and processing all the cat faces. ( ^..^)ﾉ")

parser = ImageParser()
train_images = parser.images #parser.create_batches()

image_config = pickle.load(open('image_config', 'rb'))
n_channels = image_config['CHANNELS']

print(n_channels)

gen = Generator(n_channels=n_channels).generator
disc = Discriminator(n_channels=n_channels).discriminator

gen.summary()
disc.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

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
                                 generator=gen,
                                 discriminator=disc)

EPOCHS = 300000
noise_dimmension = 100
number_of_examples_to_generate = 16


@tf.function
def train_step(images):

    noise = tf.random.normal([64, noise_dimmension])

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

        generated_images = gen(noise, training=True)

        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_grads = generator_tape.gradient(gen_loss, gen.trainable_variables)
    discriminator_grads = discriminator_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grads, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, disc.trainable_variables))

    return gen_loss, disc_loss

seed = tf.random.normal([number_of_examples_to_generate, noise_dimmension])
def train(dataset, epochs):
    gen_losses, disc_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        gen_loss, disc_loss = 0, 0

        for image_batch in dataset:
            gl, dl = train_step(image_batch)
            gen_loss += gl
            disc_loss += dl

        gen_losses.append(gen_loss/256)
        disc_losses.append(disc_loss/256)

        display.clear_output(wait=True)

        generate_and_save_images(gen, epoch + 1, seed)

        print(f'''generator loss: {gen_losses[-1]}\n''')
        print(f'''discriminator loss: {disc_losses[-1]}\n''')

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'''time for epoch {epoch + 1} is {time.time() - start}''')

    display.clear_output(wait=True)
    generate_and_save_images(gen, epoch + 1, seed)

    return (gen_losses, disc_losses)

def plot_losses(gen_losses, disc_losses, epochs=EPOCHS):
    plt.plot(range(epochs), gen_losses, label='Generator')
    plt.plot(range(epochs), disc_losses, label='Discriminator')
    plt.legend()
    plt.show()


def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig(f'''output/image_at_epoch_{epoch}.png''')
    plt.show()

gen_loss, disc_loss = train(train_images, EPOCHS)

disc.save('discriminator.h5')
gen.save('generator.h5')

plot_losses(gen_loss, disc_loss)


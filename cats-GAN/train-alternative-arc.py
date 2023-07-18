import os
import time
import config
import matplotlib.pyplot as plt
import tensorflow as tf

from image_parser import ImageParser
from discriminator import Discriminator
from generator import Generator

from IPython import display

BATCH_SIZE = config.BATCH_SIZE

print("( ^..^)ﾉ opening and processing all the cat faces. ( ^..^)ﾉ")

parser = ImageParser()
train_images = parser.create_batches()

disc, gen = Discriminator().discriminator, Generator().generator

def real_samples(image_batch):
    return image_batch, tf.ones_line(image_batch)

def latent_vector(latent_dim = 100, batch_size = config.BATCH_SIZE):
    latent_input = tf.random.normal(latent_dim * batch_size)
    latent_input = latent_input.reshape((batch_size, latent_dim))
    return latent_input

def fake_samples(gen, latent_dim=100, batch_size=config.BATCH_SIZE):

    latent_output = latent_vector(latent_dim, batch_size)
    X = gen.predict(latent_output)
    y = tf.zeros_like((batch_size, 1))
    return X, y


def def_gan(g, d):

    d.trainable = False

    model = tf.keras.Sequential(name='DCGAN-V2')
    model.add(g)
    model.add(d)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    return model

gan_model = def_gan(gen, disc)
gan_model.summary()

def train(gen, disc, dataset, latent_dim):

    for epoch in range(config.EPOCHS):

        for image_batch in dataset:

            #first we train the discriminator:
            y_real = tf.ones_like(image_batch)

            discriminator_loss, _ = disc.train_on_batch()
            pass




import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Discriminator:
    def __init__(self):
        self.discriminator = self.__make_discriminator__()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def __make_discriminator__(self):

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=[64, 64, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
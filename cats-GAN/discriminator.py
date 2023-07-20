import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Discriminator:
    def __init__(self, n_channels=1):
        self.channels = n_channels
        self.discriminator = self.__make_discriminator_2__()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def __make_discriminator_2__(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(16, (6, 6), strides=(2, 2), padding='same', input_shape=[64, 64, self.channels]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(32, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def __make_discriminator__(self):

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=[64, 64, self.channels]))
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
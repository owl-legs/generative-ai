import tensorflow as tf
from tensorflow.keras import layers

class Discriminator:
    def __init__(self):
        self.discriminator = self.__make_discriminator__()

    def __make_discriminator__(self):

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def __discriminator_loss__(self):
        return None
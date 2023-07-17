import tensorflow as tf
from tensorflow.keras import layers

class Generator:

    def __init__(self):
        self.generator = self.__make_generator__()
    def __make_generator__(self):

        model = tf.keras.Sequential()
        model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((16, 16, 256)))

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        assert model.output_shape == (None, 64, 64, 1)

        return model

    def __generator_loss__(self):
        return None
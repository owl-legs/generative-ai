import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:

    def __init__(self):
        self.generator = self.__make_generator_2__()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def __make_generator_2__(self):
        latent_dim = 100
        model = tf.keras.Sequential(name="Generator")  # Model

        # Hidden Layer 1: Start with 8 x 8 image
        n_nodes = 8 * 8 * 256  # number of nodes in the first hidden layer
        model.add(layers.Dense(n_nodes, input_dim=latent_dim, name='Generator-Hidden-Layer-1'))
        model.add(layers.Reshape((8, 8, 256), name='Generator-Hidden-Layer-Reshape-1'))

        # Hidden Layer 2: Upsample to 16 x 16
        model.add(layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-2'))
        model.add(layers.ReLU(name='Generator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3: Upsample to 32 x 32
        model.add(layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-3'))
        model.add(layers.ReLU(name='Generator-Hidden-Layer-Activation-3'))

        # Hidden Layer 4: Upsample to 64 x 64
        model.add(layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-4'))
        model.add(layers.ReLU(name='Generator-Hidden-Layer-Activation-4'))

        # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have only 1 channel)
        model.add(layers.Conv2D(filters=1, kernel_size=(5, 5), activation='tanh', padding='same',
                                name='Generator-Output-Layer'))
        return model
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

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

gen = Generator()
gen.generator.summary()
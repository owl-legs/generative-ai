import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:

    def __init__(self, n_channels=1):
        self.latent_dim = 100
        self.n_channels = n_channels
        self.generator = self.__make_generator_3__()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def __make_generator_3__(self):
        model = tf.keras.Sequential(name="Generator")  # Model

        n_nodes = 1 * 1 * 256  # number of nodes in the first hidden layer
        model.add(layers.Dense(n_nodes, input_dim=self.latent_dim, name='Generator-Hidden-Layer-1'))
        model.add(layers.Reshape((1, 1, 256), name='Generator-Hidden-Layer-Reshape-1'))

        #upsample to 2x2
        model.add(layers.Conv2DTranspose(filters=2048, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias= False,
                                         name='Generator-Hidden-Layer-2'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-2'))

        #upsample to 4x4
        model.add(layers.Conv2DTranspose(filters=1024, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias=False,
                                         name='Generator-Hidden-Layer-3'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-3'))

        #upsample to 8x8
        model.add(layers.Conv2DTranspose(filters=512, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias=False,
                                         name='Generator-Hidden-Layer-4'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-4'))

        #upsample to 16x16
        model.add(layers.Conv2DTranspose(filters=256, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias=False,
                                         name='Generator-Hidden-Layer-5'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-5'))

        #upsample to 32x32
        model.add(layers.Conv2DTranspose(filters=128, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias=False,
                                         name='Generator-Hidden-Layer-6'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-6'))

        #upsample to 64x64
        model.add(layers.Conv2DTranspose(filters=64, kernel_size=(6, 6), strides=(2, 2), padding='same',
                                         use_bias= False,
                                         name='Generator-Hidden-Layer-7'))
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-7'))

        #output image: 3 x 64 x 64
        model.add(layers.Conv2D(filters=self.n_channels, kernel_size=(6, 6),
                                activation='tanh',
                                padding='same',
                                use_bias=False,
                                name='Generator-Output-Layer'))
        return model

    def __make_generator_2__(self):

        model = tf.keras.Sequential(name="Generator")  # Model

        # Hidden Layer 1: Start with 8 x 8 image
        n_nodes = 8 * 8 * 256  # number of nodes in the first hidden layer
        model.add(layers.Dense(n_nodes, input_dim=self.latent_dim, name='Generator-Hidden-Layer-1'))
        model.add(layers.Reshape((8, 8, 256), name='Generator-Hidden-Layer-Reshape-1'))

        # Hidden Layer 2: Upsample to 16 x 16
        model.add(layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-2'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3: Upsample to 32 x 32
        model.add(layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-3'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-3'))

        # Hidden Layer 4: Upsample to 64 x 64
        model.add(layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                         name='Generator-Hidden-Layer-4'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name='Generator-Hidden-Layer-Activation-4'))

        # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have only 1 channel)
        model.add(layers.Conv2D(filters=self.n_channels, kernel_size=(5, 5), activation='sigmoid', padding='same',
                                name='Generator-Output-Layer'))
        return model
    def __make_generator__(self):

        model = tf.keras.Sequential()
        model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((16, 16, 256)))

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.n_channels, (5, 5), strides=(2, 2), use_bias=False, padding='same'))
        assert model.output_shape == (None, 64, 64, self.n_channels)

        return model

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

gen = Generator()
gen.generator.summary()
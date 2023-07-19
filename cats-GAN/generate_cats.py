import tensorflow as tf
import config
import matplotlib.pyplot as plt

generator = tf.keras.models.load_model('generator.h5')
latent_dim = config.LATENT_DIM

seed = tf.random.normal([1, latent_dim])

cat = generator(seed)

plt.imshow(cat[0, :, :, 0] * 127.5 + 127.5)
plt.show()

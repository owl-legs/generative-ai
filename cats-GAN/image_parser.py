import config
import numpy as np
import tensorflow as tf
import glob
import pickle

class ImageParser:
  def __init__(self, dir='cats/*.jpg'):
    self.images = self.__load_image_directory__()
    self.__write_image_config__()

  def __load_image_directory__(self, dirStr='cats/*.jpg'):
    images = []
    for filename in sorted(glob.glob(dirStr)):
      print(f'''loading {filename}''')
      img = tf.io.read_file(filename)
      img = tf.io.decode_image(img, channels=3)
      img = tf.image.rgb_to_grayscale(img)
      print(f'''{filename} dimmenions: {img.shape}''')
      images.append(img)
    images = np.array(images)
    return images

  def __write_image_config__(self):
    self.BUFFER_SIZE = ((self.images.shape[0]) // config.BATCH_SIZE) * config.BATCH_SIZE
    self.IMG_HEIGHT, self.IMG_WIDTH = self.images.shape[1], self.images.shape[2]
    self.CHANNELS = self.images.shape[3]
    pickle.dump({'BUFFER_SIZE':self.BUFFER_SIZE,
                 'IMG_HEIGHT':self.IMG_HEIGHT,
                 'IMG_WIDTH':self.IMG_WIDTH,
                 'CHANNELS':self.CHANNELS}, open('image_config', 'wb'), True)

  def create_batches(self):
    return tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(config.BATCH_SIZE)

#image_parser = ImageParser()

'''train_ds = tf.keras.utils.image_dataset_from_directory(
                  'cats',
                  validation_split=0.2,
                  subset="training",
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=config.BATCH_SIZE)'''



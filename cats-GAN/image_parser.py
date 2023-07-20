import config
import numpy as np
import tensorflow as tf
import glob
import pickle
import matplotlib.pyplot as plt

class ImageParser:
  def __init__(self, dir='cats/*.jpg'):
    self.images = self.__load_cats__()
    self.__write_image_config__()

  def __load_cats__(self, dirStr='cats/*.jpg'):
    cats = tf.data.Dataset.list_files(dirStr)
    cats = cats.map(lambda x: tf.image.decode_jpeg(tf.io.read_file(x)))
    cats = cats.map(lambda x: tf.image.resize(x, (64, 64)) / 255)
    cats = cats.batch(256)
    cats = cats.prefetch(tf.data.AUTOTUNE)
    return cats
  def __load_image_directory__(self, dirStr='cats/*.jpg'):
    images = []
    for filename in sorted(glob.glob(dirStr)):
      print(f'''loading {filename}''')
      img = tf.io.read_file(filename)
      img = tf.io.decode_image(img, channels=3)
      #img = tf.image.rgb_to_grayscale(img) #r(ih, iw, 3) => r(ih, iw, 1)
      print(f'''{filename} dimmenions: {img.shape}''')
      images.append(img)
    images = np.array(images).astype('float32')
    #images = (images - 127.5)/127.5
    images = images/255.0
    return images[:config.BUFFER_SIZE]

  def plot_example_cat(self):
    pass


  def __write_image_config__(self):
    #self.BUFFER_SIZE = config.BUFFER_SIZE
    #self.IMG_HEIGHT, self.IMG_WIDTH = self.images.shape[1], self.images.shape[2]
    #self.CHANNELS = self.images.shape[3]
    pickle.dump({'BUFFER_SIZE':1000000,
                 'IMG_HEIGHT':64,
                 'IMG_WIDTH':64,
                 'CHANNELS':3}, open('image_config', 'wb'), True)

  def create_batches(self):
    return tf.data.Dataset.from_tensor_slices(self.images).shuffle(self.BUFFER_SIZE).batch(config.BATCH_SIZE)

image_parser = ImageParser()
#cat_faces = image_parser.create_batches()

'''train_ds = tf.keras.utils.image_dataset_from_directory(
                  'cats',
                  validation_split=0.2,
                  subset="training",
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=config.BATCH_SIZE)'''



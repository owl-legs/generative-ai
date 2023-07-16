import glob
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)

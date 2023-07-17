from image_parser import ImageParser
from discriminator import Discriminator
from generator import Generator
import config

parser = ImageParser()
train_images = parser.create_batches()

disc, gen = Discriminator().discriminator, Generator().generator
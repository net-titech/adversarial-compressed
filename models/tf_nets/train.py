from sys import argv
from alexnet import AlexNet
import imagenet as im

# TODO: Refactoring traing code

batch_size = 256

simple_alex = AlexNet(batch_size, init_lr=0.05)
data_generator = im.gen_data
simple_alex.train(data_generator, im.train_size, epoch=100)

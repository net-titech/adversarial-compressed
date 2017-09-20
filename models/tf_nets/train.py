from sys import argv
from alexnet import AlexNet, AlexNetVD
import imagenet as im

# TODO: Refactoring traing code

batch_size = 256
data_generator = im.gen_data

#simple_alex = AlexNet(batch_size, init_lr=0.1)
#simple_alex.train(data_generator, im.train_size, epoch=200, step_save=10000)

vd_fc_alex = AlexNetVD(init_alpha=0.999, batch_size=batch_size, init_lr=0.1)
vd_fc_alex.train(data_generator, im.train_size, epoch=30, step_save=10001)

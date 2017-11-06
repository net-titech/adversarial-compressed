from sys import argv
#from alexnet import AlexNet
#from alexnet_vd import AlexNetVD
#import imagenet as im

# TODO: Refactoring traing code

#batch_size = 256
#data_generator = im.gen_data

#simple_alex = AlexNet(batch_size, init_lr=0.1)
#simple_alex.train(data_generator, im.train_size, epoch=200, step_save=10000)

#vd_fc_alex = AlexNetVD(init_alpha=0.999, batch_size=batch_size, init_lr=0.1)
#vd_fc_alex.train(data_generator, im.train_size, epoch=30, step_save=10001)

#l2_alex = AlexNet(batch_size, init_lr=0.05, name="AlexNet_L2")
#l2_alex.train(data_generator, im.train_size, epoch=200, step_save=10000)

# batch_size=512 - 20170928 - failed
#vd_fc_alex = AlexNetVD(init_alpha=1e-8, batch_size=batch_size*2,
#                       regularization_weight=1.0,
#                       init_lr=0.0001, name="AlexNet_VDFC")

#vd_fc_alex = AlexNetVD(init_alpha=1e-2, batch_size=batch_size,
#                       regularization_weight=0.1,
#                       init_lr=0.01, name="AlexNet_VDFC_REG_SCALED")
#try:
#    vd_fc_alex.train(data_generator, im.train_size, epoch=30, step_save=10000)
#except KeyboardInterrupt:
#    print("Ended training. Dumping weights.")
#    # TODO: Impl this

from mnist import load_mnist
from lenet import LeNet
mnist_data = load_mnist(loc="/mnt/data/mnist")
default_lenet = LeNet(name="default_lenet")
default_lenet.train(mnist_data, epoch=30)

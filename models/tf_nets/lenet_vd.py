import tensorflow as tf
from utils import log_training
from collections import namedtuple
from variational_layers import convVD, denseVD, vd_reg
from lenet import LeNet, caffe_inv_decay

Preds = namedtuple("Predictions", ["label", "prob"])

class LeNet_VDFC(LeNet):
    """
        BLVC LeNet (modified version w.o. learning rate mult)
        with variational dropout added to the fully connected layers
    """
    def __init__(self, train_batch_size=64, train_size=55000,
                 test_batch_size=100, val_size=5000,
                 image_size=28, image_channels=1,
                 num_classes=10, init_lr=0.01,
                 momentum=0.9, gamma=1e-5, lr_decay_step=1,
                 power=0.75, l2_scale=5e-4, vdrw=0.1,
                 name="LeNet_VDFC", summary_dir="./"):
        super(self).__init__()
        self.batch_size = train_batch_size
        self.train_size = train_size  # Need to know data size for SGVD
        self.val_size = val_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.lr = init_lr
        self.gamma = gamma # lr decay factor
        self.power = power # inv learning policy param
        self.sum_dir = summary_dir
        self.built = False
        self.l2_scale = l2_scale
        self.momentum = momentum
        self.lr_decay_step = lr_decay_step
        self.vdrw = vdrw  # D_LK regularization strength
        self.name = name

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.input = tf.placeholder(tf.float32, shape=[None,
                                                           self.image_size,
                                                           self.image_size,
                                                           self.image_channels],
                                         name="input_images")
            self.labels = tf.placeholder(tf.int32, shape=[None],
                                         name="train_labels")
        with tf.name_scope("settings"):
            self.training = tf.placeholder(tf.bool, shape=(), name="is_training")

    def _create_net(self):
        with tf.name_scope("convolution_group"):
            # conv1 5x5x20
            conv1 = tf.layers.conv2d(inputs=self.input, filters=20, kernel_size=5,
                                     strides=1, padding="valid", name="conv1")
            maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2,
                                               strides=2, padding="valid",
                                               name="maxpool1")
            # conv2 5x5x50
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=50, kernel_size=5,
                                     strides=1, padding="valid", name="conv2")
            maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2,
                                               strides=2, padding="valid",
                                               name="maxpool2")
        with tf.name_scope("fully_connected_group"):
            # fc1 500 units
            flat = tf.contrib.layers.flatten(maxpool2)
            fc1_vd, vdl1 = denseVD(inputs=flat, units=500,
                                   activation=tf.nn.relu, name="vdfc1")
            fc2 = tf.layers.dense(inputs=fc1_vd, units=10,
                                  activation=tf.nn.relu, name="fc2")
            self.logits = fc2
        self.vd_layers = [vdl1]  #TODO: Fix hardcode
        with tf.name_scope("output"):
            self.predictions = Preds(label=tf.argmax(input=self.logits, axis=1),
                                   prob=tf.nn.softmax(self.logits,
                                       name="softmax"))
        with tf.name_scope("accuracy"):
            acc_top1 = tf.nn.in_top_k(self.logits, self.labels, 1)
            acc_top1 = tf.cast(acc_top1, tf.float32)
            self.acc1 = tf.reduce_mean(acc_top1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            onehot_labels = tf.one_hot(indices=self.labels,
                                       depth=self.num_classes)
            loss = tf.losses.softmax_cross_entropy(onehot_labels,
                                        self.logits,
                                        reduction=tf.losses.Reduction.SUM)
            loss = (self.train_size / self.batch_size * 1.0) * loss
            if self.l2_scale:
                weights = [var for var in tf.global_variables()\
                           if r"/kernel:" in var.name]
                print("Debug: ", weights)
                l2_term = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
                loss += self.l2_scale * l2_term
            vd_term = tf.reduce_sum([vd_reg(l.alpha) for l in self.vd_layers])
            self.loss = loss + self.vdrw * vd_term

    def _create_optimizer(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name="global_step")
        with tf.name_scope("trainer"):
            lr = caffe_inv_decay(self.lr, self.global_step,
                                 self.lr_decay_step, self.gamma, self.power)
            self.optimizer = tf.train.MomentumOptimizer(lr, self.momentum)
            self.train_op = self.optimizer.minimize(self.loss,
                                            global_step=self.global_step)

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
        super().__init__()
        self.batch_size = train_batch_size
        self.train_size = train_size  # Need to know data size for SGVD
        self.val_size = val_size
        self.test_size = test_batch_size
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
            self.input = tf.placeholder(tf.float32, shape=[self.batch_size,
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
            fc1_vd, vdl1 = denseVD(inputs=flat, training=self.training, 
                                   units=500, activation=tf.nn.relu, 
                                   name="vdfc1")
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

    def _build(self, tfgraph=None):
        if tfgraph is None:
            tfgraph = tf.get_default_graph()
        self.graph = tfgraph
        with tfgraph.as_default():
            self._create_placeholders()
            self._create_net()
            self._create_loss()
            super()._create_optimizer()
            super()._create_summary()
        self.built = True

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

    def train(self, data_gen, epoch=1, continue_from=None,
              step_save=5000, step_val=1000, step_log=100):
        if not self.built:
            self._build()
        opts = tf.ConfigProto(allow_soft_placement=True)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=opts, graph=self.graph) as sess:
            if continue_from:
                print("Continue training from " + continue_from)
                saver.restore(sess, continue_from)
            else:
                sess.run(init_op)
            writer_train = tf.summary.FileWriter("./logs/train", sess.graph)
            writer_val = tf.summary.FileWriter("./logs/val", sess.graph)
            while data_gen.train.epochs_completed < epoch:
                images, labels = data_gen.train.next_batch(self.batch_size)
                gl_step = self.global_step.eval()
                if gl_step % step_val == 0:
                    acc = sess.run([self.acc1], feed_dict={
                        self.input: images, self.labels: labels,
                        self.training: False})
                    print("Step {}, training acc: {}".format(gl_step, acc))
                    val_images, val_labels = data_gen.val.next_batch(self.batch_size)
                    acc = sess.run([self.acc1], feed_dict={
                        self.input: val_images, self.labels: val_labels,
                        self.training: False})
                    print("Step {}, val acc: {}".format(gl_step, acc))
                _, s, gl_step, bloss, lr = sess.run(
                    [self.train_op, self.summary_ops, self.global_step,
                     self.loss, self.optimizer._learning_rate_tensor],
                    feed_dict={self.input: images, self.labels: labels,
                               self.training: True})
                writer_train.add_summary(s, global_step=gl_step)
                if gl_step % step_log == 0:
                    log_training(gl_step, bloss, lr)
                if gl_step % step_save == 0 and gl_step > 0:
                    print("Saving checkpoint...")
                    saver.save(sess, "./checkpoints/{}".format(self.name),
                                global_step=gl_step)
            test_images, test_labels = data_gen.text.next_batch(64)
            acc = sess.run([self.acc1], feed_dict={self.input: test_images,
                                                   self.labels: test_labels,
                                                   self.training: False})
            print("Step {}, 64 samples test acc: {}".format(gl_step, acc))

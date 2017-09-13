import tensorflow as tf

class AlexNet:
    """
    BLVC AlexNet (Single stream version)
    Paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks 
    """
    def __init__(self, batch_size, image_size=227, image_channels=3, 
                 num_classes=1000, init_lr=0.1, stepsize=100000, 
                 gamma=0.1):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.lr = init_lr
        self.lr_decay_step = stepsize
        self.gamma = gamma # lr decay factor

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.input = tf.placeholder(tf.float32, shape=[self.batch_size, 
                                                           self.image_size, 
                                                           self.image_size,
                                                           self.image_channels], 
                                         name="input_images")
            self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1],
                                         name="train_labels")
        with tf.name_scope("settings"):
            self.training = tf.placeholder(tf.bool, shape=(), name="is_training")

    def _create_net(self):
        with tf.name_scope("convolution_group"):
            # conv1 11x11x96
            conv1 = tf.layers.conv2d(inputs=self.input, filters=96,
                                     kernel_size=11, strides=4,
                                     padding='same', activation=tf.nn.relu,
                                     name="conv1")
            lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, alpha=0.0001, 
                             beta=0.75, name="lrn1")
            maxpool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=3,
                                               strides=2, padding="valid",
                                               name="maxpool1")
            # conv2 5x5x256
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=256,
                                     kernel_size=5, strides=1,
                                     padding='same', activation=tf.nn.relu,
                                     name="conv2")
            lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, alpha=0.0001, 
                             beta=0.75, name="lrn1")
            maxpool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=3,
                                               strides=2, padding="valid",
                                               name="maxpool2")
            # conv3 3x3x384
            conv3 = tf.layers.conv2d(inputs=maxpool2, filters=384,
                                     kernel_size=3, strides=1,
                                     padding="same", activation=tf.nn.relu,
                                     name="conv3")
            # conv4 3x3x384
            conv4 = tf.layers.conv2d(inputs=conv3, filters=384,
                                     kernel_size=3, strides=1,
                                     padding="same", activation=tf.nn.relu,
                                     name="conv4")
            # conv5 3x3x256
            conv5 = tf.layers.conv2d(inputs=conv4, filters=256,
                                     kernel_size=3, strides=1,
                                     padding="same", activation=tf.nn.relu,
                                     name="conv5")
            maxpool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3,
                                               strides=2, padding="valid",
                                               name="maxpool5")
        with tf.name_scope("fully_connected_group"): 
            # fc6 4096 units
            flat5 = tf.contrib.layers.flatten(maxpool5)
            fc6 = tf.layers.dense(inputs=flat5, units=4096, 
                                  activation=tf.nn.relu, name="fc6")
            dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, 
                                         training=self.training,
                                         name="dropout6")
            fc7 = tf.layers.dense(inputs=dropout6, units=4096, 
                                  activation=tf.nn.relu, name="fc7")
            dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, 
                                         training=self.training,
                                         name="dropout7")
            self.logits = tf.layers.dense(inputs=dropout7, 
                                          units=self.num_classes, name="fc8")
        with tf.name_scope("output"):
            self.predictions = {
                "top1": tf.argmax(input=self.logits, axis=1),
                "probs": tf.nn.softmax(self.logits, name="softmax")
            } 
            self.acc_top1 = tf.nn.in_top_k(self.logits, labels, 1)
            self.acc_top5 = tf.nn.in_top_k(self.logits, labels, 5)

    def _create_loss(self):
        with tf.name_scope("loss"):
            onehot_labels = tf.one_hot(indices=self.labels,
                                       depth=self.num_classes)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels, 
                                                        self.logits)

    def _create_optimizer(self):
        # Global step for learning rate deca
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name="global_step")
        with tf.name_scope("trainer"):
            lr = tf.train.exponential_decay(self.lr, self.global_step,
                                                       self.lr_decay_step, self.gamma)
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
            self.train_op = self.optimizer.minimize(self.loss, 
                                                    global_step=self.global_step) 

    def _create_summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_saver(self):
        pass

    def build(self, tfgraph=None):
        if tfgraph is None:
            tfgraph = tf.get_default_graph()
        with tfgraph.as_default():
            self._create_placeholders()
            self._create_net()
            self._create_loss()
            self._create_optimizer()
            self._create_summary()
            self._create_saver()


class AlexNetVD(AlexNet):
    """
    AlexNet with Variational Dropout
    Paper: (Kingma, 2015)
    """
    pass


class AlexNetSVD(AlexNet):
    """
    AlexNet with Sparse Variational Dropout
    Paper: (Molchanov, 2017)
    df""

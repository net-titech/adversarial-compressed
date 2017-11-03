import tensorflow as tf
from alexnet import AlexNet
from variational_layers import denseVD, convVD, vd_reg

class AlexNetSpVD(AlexNet):
    """
    AlexNet with Sparse Variational Dropout
    Paper: (Molchanov, 2015)
    """
    def __init__(self, init_alpha=0.5, num_samples=1281167,
                 regularization_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha
        self.name="AlexNetVD"
        self.regularization_weight = regularization_weight
        self.num_samples = num_samples
    
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
            # fc6 4096 units with variational dropout
            flat5 = tf.contrib.layers.flatten(maxpool5)
            fc6, lvdfc6 = denseVD(inputs=flat5, units=4096, training=self.training,
                          activation=tf.nn.relu, name="vdfc6")
            # fc7 4096 units with variational dropout
            fc7, lvdfc7 = denseVD(inputs=fc6, units=4096, training=self.training,
                                    activation=tf.nn.relu, name="vdfc7")
            # fc8 is also logits with variational dropout
            self.logits, lvdfc8 = denseVD(inputs=fc7, training=self.training,
                                          units=self.num_classes, name="vdfc8")
        
        self.vd_layers = [lvdfc6, lvdfc7, lvdfc8]

        with tf.name_scope("output"):
            self.predictions = {
                "class": tf.argmax(input=self.logits, axis=1),
                "probs": tf.nn.softmax(self.logits, name="softmax")
            } 
        with tf.name_scope("accuracy"):
            acc_top1 = tf.nn.in_top_k(self.logits, self.labels, 1)
            acc_top5 = tf.nn.in_top_k(self.logits, self.labels, 5)
            acc_top1 = tf.cast(acc_top1, tf.float32)
            acc_top5 = tf.cast(acc_top5, tf.float32)
            self.acc1 = tf.reduce_mean(acc_top1)
            self.acc5 = tf.reduce_mean(acc_top5)

    def _create_loss(self):
        num_samples = self.num_samples
        rw = self.regularization_weight
        with tf.name_scope("loss"):
            onehot_labels = tf.one_hot(indices=self.labels,
                                       depth=self.num_classes)
            ell = -tf.reduce_sum(tf.losses.softmax_cross_entropy(onehot_labels,
                                                                 self.logits))
            reg = tf.reduce_sum([vd_reg(l.get_alpha()) for l in self.vd_layers])
            self.loss = -((num_samples * 1.0 / self.batch_size)*ell - rw * reg)



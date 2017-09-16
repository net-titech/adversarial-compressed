import tensorflow as tf
from utils import log_training

class AlexNet:
    """
    BLVC AlexNet (Single stream version)
    Paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks 
    """
    def __init__(self, batch_size, image_size=227, image_channels=3, 
                 num_classes=1000, init_lr=0.1, stepsize=100000, 
                 gamma=0.1, summary_dir="./"):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.lr = init_lr
        self.lr_decay_step = stepsize
        self.gamma = gamma # lr decay factor
        self.sum_dir = summary_dir
        self.built = False
        self.name = "AlexNet"

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
        with tf.name_scope("loss"):
            onehot_labels = tf.one_hot(indices=self.labels,
                                       depth=self.num_classes)
            loss = tf.losses.softmax_cross_entropy(onehot_labels, 
                                                   self.logits)
            self.loss = tf.reduce_mean(loss)

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
            tf.summary.scalar("top1_accuracy", self.acc1)
            tf.summary.scalar("top5_accuracy", self.acc5)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_ops = tf.summary.merge_all()

    def _build(self, tfgraph=None):
        if tfgraph is None:
            tfgraph = tf.get_default_graph()
        self.graph = tfgraph
        with tfgraph.as_default():
            self._create_placeholders()
            self._create_net()
            self._create_loss()
            self._create_optimizer()
            self._create_summary()
        self.built = True
    
    def visualize_graph(self, folder="./graphs"):
        """
        [===in python or ipython===]
        test = AlexNet(100)
        test.visualize_graph() 
        [===in shell===]
        $ tensorboard --logdir="./graphs"
        [===in browser===]
        GOTO: http://<machine_ip>:6006/#graphs
        """
        if not self.built:
            self._build()
        with tf.Session(graph=self.graph) as sess:
            writer = tf.summary.FileWriter(folder, sess.graph)
        writer.close()

    def train(self, data_gen, data_size, epoch=1, continue_from=None, 
              step_save=10000, step_log=100, step_training_log=20):
        if not self.built:
            self._build()
        opts = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=True)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=opts, graph=self.graph) as sess:
            # Create session
            if continue_from is not None:
                saver.restore(sess, continue_from)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./logs", sess.graph)
            images_labels = data_gen(self.batch_size, "train")
            val_images_labels = data_gen(self.batch_size, "val")
            # Generate data and training
            for e in range(epoch):
                print("======== Epoch {} ========".format(e))
                for _ in range(data_size//self.batch_size):
                    images, labels = next(images_labels)
                    gl_step = self.global_step.eval()
                    if gl_step % step_log == 0:
                        # Training accuracies
                        acc1, acc5 = sess.run([self.acc1, self.acc5], feed_dict={
                            self.input: images, self.labels: labels, 
                            self.training: False})
                        print("Step {}, training accuracy: {} (top 1), {} (top 5)".\
                            format(gl_step, acc1, acc5))
                        # Validation accuracies
                        val_images, val_labels = next(val_images_labels)
                        acc1, acc5 = sess.run([self.acc1, self.acc5], feed_dict={
                            self.input: val_images, self.labels: val_labels, 
                            self.training: False})
                        print("Step {}, validation accuracy: {} (top 1), {} (top 5)".\
                            format(gl_step, acc1, acc5))
                    # Training
                    _, s, gl_step, bloss = sess.run([self.train_op, self.summary_ops, 	
                        self.global_step, self.loss], feed_dict={self.input: images,
                        self.labels: labels, self.training: True})
                    writer.add_summary(s, global_step=gl_step)
                    if gl_step % step_training_log == 0:
                        log_training(gl_step, bloss)
                    if (gl_step+1) % step_save == 0:
                        print("Saving checkpoint...")
                        saver.save(sess, "./checkpoints/{}".format(self.name),
                                   global_step = gl_step)
                   
class AlexNetVD(AlexNet):
    """
    AlexNet with Variational Dropout
    Paper: (Kingma, 2015)
    """
    def __init__(self, alpha, sigma, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.sigma = sigma


class AlexNetSVD(AlexNet):
    """
    AlexNet with Sparse Variational Dropout
    Paper: (Molchanov, 2017)
    """
    def __init__(self, log_alpha, log_sigma, **kwargs):
        super().__init__(**kwargs)
        self.log_alpha = log_alpha
        self.log_sigma = log_sigma


import tensorflow as tf
from utils import log_training
from collections import namedtuple

Preds = namedtuple("Predictions", ["label", "prob"])

def caffe_inv_decay(learning_rate, global_step, decay_step,
                    gamma, power, name=None):
    """
        Impement learning policy 'inv' used in Caffe:
        lr = base_lr * (1 + gamma * iter) ^ (-power)
        Test:
            import tensorflow as tf
            gs = tf.Variable(0, trainable=False)
            lr = caffe_inv_decay(base_lr, gs, 1, 1e-5, 0.75)
            sess = tf.Session()
            sess.run(tf.global_variable_intializer())
            sess.run(lr)
                0.0099998
            sess.run(tf.assign(gl, 20000))
                20000
            sess.run(lr)
                0.0087219598
    """
    if global_step is None:
        raise ValueError("global_step is required.")
    with tf.name_scope(name, "CaffeInvDecay",
                       [learning_rate, global_step, gamma]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate,
                                              name="learning_rate")
        lr_dtype = learning_rate.dtype
        global_step = tf.cast(global_step, lr_dtype)
        decay_step = tf.cast(decay_step, lr_dtype)
        gamma = tf.cast(gamma, lr_dtype)
        p = global_step / decay_step
        one_const = tf.cast(tf.constant(1), lr_dtype)
        pow_const = tf.cast(tf.constant(-power), lr_dtype)
        base = tf.add(one_const, tf.multiply(gamma, p))
        scale = tf.pow(base, pow_const)
        return tf.multiply(learning_rate, scale)

class LeNet(object):
    """
        BLVC LeNet (modified version w.o. learning rate mult)
        URL: https://goo.gl/ezxzvj
    """
    def __init__(self, train_batch_size=64,
                 test_batch_size=100, val_size=5000,
                 image_size=28, image_channels=1,
                 num_classes=10, init_lr=0.01,
                 momentum=0.9, gamma=1e-5, lr_decay_step=1,
                 power=0.75, l2_scale=5e-4,
                 name="LeNet", summary_dir="./"):
        self.batch_size = train_batch_size
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
        self.name = name

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.input = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                           self.image_size,
                                                           self.image_size,
                                                           self.image_channels],
                                         name="input_images")
            self.labels = tf.placeholder(tf.int32, shape=[self.batch_size],
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
            fc1 = tf.layers.dense(inputs=flat, units=500,
                                  activation=tf.nn.relu, name="fc1")
            fc2 = tf.layers.dense(inputs=fc1, units=10,
                    activation=tf.nn.relu, name="fc2")
            self.logits = fc2
        with tf.name_scope("output"):
            self.predictions = Preds(label=tf.argmax(input=self.logits, axis=1),
                                   prob=tf.nn.softmax(self.logits, name="softmax"))
        with tf.name_scope("accuracy"):
            acc_top1 = tf.nn.in_top_k(self.logits, self.labels, 1)
            acc_top1 = tf.cast(acc_top1, tf.float32)
            self.acc1 = tf.reduce_mean(acc_top1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            weights = [var for var in tf.global_variables() if r"/kernel:" in var.name]
            print("Debug: ", weights)
            l2_term = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
            onehot_labels = tf.one_hot(indices=self.labels,
                                       depth=self.num_classes)
            loss = tf.losses.softmax_cross_entropy(onehot_labels, self.logits)
            self.loss = tf.reduce_mean(loss)
            self.loss += self.l2_scale * l2_term

    def _create_optimizer(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name="global_step")
        with tf.name_scope("trainer"):
            lr = caffe_inv_decay(self.lr, self.global_step,
                                 self.lr_decay_step, self.gamma, self.power)
            self.optimizer = tf.train.MomentumOptimizer(lr, self.momentum)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step)

    def _create_summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("top1_accuracy", self.acc1)
            tf.summary.histogram("loss_hist", self.loss)
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
              step_save=500, step_val=100, step_log=10):
        if not self.built:
            self._build()
        opts = tf.ConfigProto(allow_soft_placement=True)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=opts, graph=self.graph) as sess:
            # Create session
            if continue_from is not None:
                print("Continue training from " + continue_from)
                saver.restore(sess, continue_from)
            else:
                sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter("./logs/train", sess.graph)
            writer_val = tf.summary.FileWriter("./logs/val", sess.graph)
            images_labels = data_gen(self.batch_size, "train")
            val_images_labels = data_gen(self.batch_size, "val")
            # Generate data and training
            while data_gen.train.epoch_completed() < epoch:
                print("======== Epoch {} ========".\
                        format(data_gen.train.epoch_completed()+1))
                for images, labels in data_gen.train.next_batch(self.batch_size):
                    gl_step = self.global_step.eval()
                    if gl_step % step_val == 0:
                        # Training accuracies
                        acc1 = sess.run([self.acc1], feed_dict={
                            self.input: images, self.labels: labels,
                            self.training: False})
                        print("Step {}, training accuracy: {} (top 1)".\
                            format(gl_step, acc1))
                        # Validation accuracies
                        val_images, val_labels = data_gen.val.next_batch(self.val_size)
                        acc1, s = sess.run([self.acc1, self.summary_ops],
                                           feed_dict={self.input: val_images,
                                                      self.labels: val_labels,
                                                      self.training: False})
                        writer_val.add_summary(s, global_step=gl_step)
                        print("Step {}, validation accuracy: {} (top 1)".\
                            format(gl_step, acc1))
                    # Training
                    _, s, gl_step, bloss, lr = sess.run(
                        [self.train_op, self.summary_ops, self.global_step,
                         self.loss, self.optimizer._learning_rate_tensor],
                        feed_dict={self.input: images,
                                   self.labels: labels,
                                   self.training: True})
                    writer_train.add_summary(s, global_step=gl_step)
                    if gl_step % step_log == 0:
                        log_training(gl_step, bloss, lr)
                    if gl_step % step_save == 0 and gl_step > 0:
                        print("Saving checkpoint...")
                        saver.save(sess, "./checkpoints/{}".format(self.name),
                                   global_step = gl_step)

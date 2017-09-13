import tensorflow as tf

class AlexNet:
    """
    BLVC AlexNet (Single stream version)
    Paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks 
    """
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        pass

    def _create_placeholders(self):
        with tf.name_scope("data"):
            input_images = tf.placeholder(tf.float32, shape=[self.batch_size, 
                                                             self.image_size, 
                                                             self.image_size], 
                                          name="input_images")
        pass

    def _create_net(self):
        pass

    def _create_loss(self):
        pass

    def _create_summary(self):
        pass

    def _create_saver(self):
        pass


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
    """

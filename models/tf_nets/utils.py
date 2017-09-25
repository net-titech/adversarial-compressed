import time
import tensorflow as tf

start = 0

def log_training(step, bloss, lr):
    global start
    if start == 0:
        start = time.time()
        print("|Start| Step {}: batch loss: {} learning rate: {}".format(step, bloss, lr))
    else:
        elapsed = time.time() - start
        print("|{0:5.2f}(sec)| Step {1}: batch loss: {2} learning rate {3}".format(elapsed, step, bloss, lr))

def convert_data_format(data_format, ndim):
    """Copy from tensorflow/tensorflow/python/layers/utils.py"""
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError("Input rank not supported:", ndim)
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError("Input rank not supported:", ndim)
    else:
        raise ValueError("Invalid data_format:", data_format)

def load_weights(ckpt_file, net_class):
    net = net_class()
    config = tf.ConfigProto(device_count={"GPU":0})
    with tf.Session(config=config) as sess:
        net._build()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)
        all_vars = {}
        for var in tf.global_variables():
            all_vars[var.name] = sess.run(var)
    return all_vars 

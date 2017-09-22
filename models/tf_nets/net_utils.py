import tensorflow as tf

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

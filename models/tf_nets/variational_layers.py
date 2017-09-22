import tensorflow as tf
from tensorflow.contrib.keras import layers

class DenseVarDrop(layers.Dense):
    """Variational Dropout for fully connected layer
    Paper: Variational Dropout and the Local Reparameterization Trick
    (Kingma, 2015)
    """
    def __init__(self, 
                 init_alpha=1.0, 
                 alpha_reg=None, 
                 use_alpha_bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha
        self.alpha_reg = alpha_reg 
        self.use_alpha_bias = use_alpha_bias

    def build(self, input_shape, dropout_mode="weights"):
        # TODO: Implement weights and units dropout modes
        input_shape = tf.TensorShape(input_shape)
        self.log_alpha = self.add_variable("log_alpha", 
                            shape=None,
                            initializer=tf.fill([input_shape[-1].value, self.units], 
                                                tf.log(self.init_alpha)),
                            regularizer=self.alpha_reg,
                            dtype=self.dtype,
                            trainable=True)
        if self.use_alpha_bias:
            self.alpha_bias = self.add_variable("alpha_bias",
                                                shape=[self.units,],
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                dtype=self.dtype,
                                                trainable=True)
        else:
            self.alpha_bias = None
        super(DenseVarDrop, self).build(input_shape)

    def call(self, inputs, training):
        def vd_dropout():
            # Variational Dropout
            tinputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tinputs.get_shape().as_list()
            output_shape = shape[:1] + [self.units]
            kernel_sq = self.kernel * self.kernel
            # Local reparameterization trick
            noise = tf.random_normal(shape=output_shape)
            alpha = tf.clip_by_value(tf.exp(self.log_alpha), 0, 1)
            if len(output_shape) > 2:
                # Broadcasting is required for the inputs
                mean = tf.tensordot(tinputs, self.kernel, [[len(shape)-1], [0]])
                var = tf.sqrt(tf.tensordot(tinputs * tinputs, 
                                           alpha * kernel_sq))
                outputs = mean + noise * var
                # Reshape the output back to the org ndim
                outputs.set_shape(output_shape)
            else:
                mean = tf.matmul(tinputs, self.kernel)
                var = tf.sqrt(tf.matmul(inputs*inputs,
                                        alpha * kernel_sq))
                outputs = mean + noise * var
            # TODO: Think about the use of alpha bias
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)
            if self.activation is not None:
                return self.activation(outputs)
            return outputs
        def mul_dense():  # TODO: Use super function
            # Default dense layer behavior  
            tinputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tinputs.get_shape().as_list()
            output_shape = shape[:1] + [self.units]
            if len(output_shape) > 2:
                # Broadcasting is required for the inputs
                outputs = tf.tensordot(tinputs, self.kernel, [[len(shape)-1], [0]])
                # Reshape the output back to the org ndim
                outputs.set_shape(output_shape)
            else:
                outputs = tf.matmul(tinputs, self.kernel)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)
            if self.activation is not None:
                return self.activation(outputs)
            return outputs
        # Switch between training and testing        
        return tf.cond(training, vd_dropout, mul_dense)

def denseVD(
    inputs,
    training,
    init_alpha=1.0,
    alpha_reg=False,
    use_alpha_bias=False,
    **kwargs):
    layer = DenseVarDrop(init_alpha, alpha_reg,
                                    use_alpha_bias, **kwargs)
    return layer.apply(inputs, training=training)

def vd_reg(alpha, constant=0.5):
    """Compute DK divergece between approximate
    posterior and the prior. This term is refered 
    as a regularization term prefering the posterior
    to be similar to the prior."""
    # TODO: Think about the constant value
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921
    return constant + 0.5*tf.log(alpha) + tf.multiply(c1,alpha) +\
           tf.multiply(c2,tf.pow(alpha,2)) + tf.multiply(c3,tf.pow(alpha,3))

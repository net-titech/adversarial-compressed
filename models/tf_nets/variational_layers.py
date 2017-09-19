import tensorflow as tf
from tensorflow.contrib.keras import layers

class DenseVariationalDropout(layers.Dense):
    """Variational Dropout for fully connected layer
    Paper: Variational Dropout and the Local Reparameterization Trick
    (Kingma, 2015)
    """
    def __init__(self, 
                 init_alpha=1.0, 
                 alpha_reg=None, 
                 use_alpha_bias=False,
                 **kwargs):
        super(DenseVariationalDropout, self).__init__(**kwargs)
        self.init_alpha = init_alpha
        self.alpha_reg = alpha_reg 
        self.use_alpha_bias = use_alpha_bias

    def build(self, input_shape, dropout_mode="weights"):
        # TODO: Implement weights and units dropout modes
        input_shape = tf.TensorShape(input_shape)
        self.alpha = self.add_variable("alpha", 
                                       shape=[input_shape[-1].value, self.units],
                                       initializer=self.init_alpha,
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
        super(DenseVariationalDropout, self).build(input_shape)
    
    def call(self, inputs, training=False):
        # Deterministic
        if not training:
            return super(DenseVariationalDropout, self).call(inputs)
        # Variational Dropout
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        output_shape = shape[:1] + [self.units]
        kernel_sq = self.kernel * self.kernel
        noise = tf.random_normal(shape=self.get_shape())
        if len(output_shape) > 2:
            # Broadcasting is required for the inputs
            mean = tf.tensordot(inputs, self.kernel, [[len(shape)-1],
                                                         [0]])
            var = tf.sqrt(tf.tensordot(inputs * inputs, 
                                       self.alpha * kernel_sq))
            outputs = mean + noise * var
            # Reshape the output back to the org ndim
            outputs.set_shape(output_shape)
        else:
            mean = tf.matmul(inputs, self.kernel)
            var = tf.sqrt(tf.matmul(inputs*inputs,
                                    self.alpha * kernel_sq)) 
            outputs = mean + noise * var
        # TODO: Think about the use of alpha bias
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def denseVD(
    inputs,
    init_alpha,
    alpha_reg,
    use_alpha_bias,
    **kwargs):
    layer = DenseVariationalDropout(init_alpha, alpha_reg,
                                    use_alpha_bias, **kwargs)
    return layer.apply(inputs)

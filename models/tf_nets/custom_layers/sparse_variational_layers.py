import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow import nn
from utils import convert_data_format

eps = 1e-8 # Keep away from log(0)

class ConvSpVarDrop(layers.Conv2D):
    """Sparse Variational Dropout for convolutional layer
    https://github.com/gear/variational-dropout-sparsifies-dnn/blob/master/nets/layers.py
    (Molchanov, 2017)
    """
    def __init__(self,
                 init_log_sigma2=-10,
                 log_sigma2_reg=None,
                 use_log_sigma2_bias=False,
                 name="SpVarDropConv",
                 **kwargs):
        super(ConvSpVarDrop, self).__init__(**kwargs)
        self.init_log_sigma2 = init_log_sigma2
        self.log_sigma2_reg = log_sigma2_reg
        self.use_log_sigma2_bias = use_log_sigma2_bias
        self.name = name

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError("The channel dimension of the inputs "
                             "should be defined. Found `None`.")
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.log_sigma2 = self.add_variable(name="log_sigma2",
                                            shape=None,
                                            initializer=tf.fill(kernel_shape, 
                                                                tf.log(self.init_log_sigma2)),
                                            regularizer=self.log_sigma2_reg,
                                            dtype=self.dtype,
                                            trainable=True) 
        if self.use_log_sigma2_reg:
            print("Not implemented. Does it make sense?")
        super(ConvSpVarDrop, self).build(input_shape)

    def call(self, inputs, training, thres=3, train_prune=False):
        zeros = tf.zeros_like(self.kernel)
        log_alpha = tf.clip_by_value(self.log_sigma2 - tf.log(eps+self.kernel*self.kernel), -8, 8)
        kernel_prune_mask = tf.greater_equal(log_alpha, thres)
        def conv_sp_var_drop():
            input2 = tf.multiply(inputs,inputs)
            # Clip value as suggested by the author's implementation
            if train_prune:
                kern = tf.select(kernel_prune_mask, self.kernel, kernel_clip_mask)
            # Convolution with reparameterization trick
            theta = nn.convolution(input=inputs,
                         filter=kern,
                         dilation_rate=self.dilation_rate,
                         strides=self.strides,
                         padding=self.padding.upper(),
                         data_format=convert_data_format(self.data_format, self.rank+2))
            sigma = tf.sqrt(nn.convolution(input=input2,
                                 filter=tf.exp(log_alpha)*kern*kern,
                                 dilation_rate=self.dilation_rate,
                                 strides=self.strides,
                                 padding=self.padding.upper(),
                                 data_format=convert_data_format(self.data_format, self.rank+2)))
            noise = tf.random_normal(shape=theta.shape.as_list())
            outputs = theta + noise * sigma
            # bias
            if self.bias is not None:
                if self.data_format == "channels_first":
                    if self.rank == 1:
                        bias = tf.reshape(self.bias, (1, self.filters, 1))
                        outputs += bias
                    if self.rank == 2:
                        outputs = tf.nn.bias_add(outputs, self.bias, data_format="NCHW")
                    if self.rank == 3:
                        outputs_shape = outputs.shape.as_list()
                        outputs_4d = tf.reshape(outputs,
                                                [outputs_shape[0], outputs_shape[1],
                                                 outputs_shape[2] * outputs_shape[3],
                                                 otuputs_shape[4]])
                        outputs_4d = tf.nn.bias_add(outputs_4d, self.bias, data_format="NCHW")
                        outputs = tf.reshape(outputs_4d, outputs_shape)
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")
            # Activation
            if self.activation is not None:
                return self.activation(outputs)
            return outputs

        def deterministic():
            kern = tf.select(kernel_prune_mask, zeros, self.kernel)
            # Convolution with reparameterization trick
            outputs = nn.convolution(input=inputs,
                         filter=kern,
                         dilation_rate=self.dilation_rate,
                         strides=self.strides,
                         padding=self.padding.upper(),
                         data_format=convert_data_format(self.data_format, self.rank+2))
            # bias
            if self.bias is not None:
                if self.data_format == "channels_first":
                    if self.rank == 1:
                        bias = tf.reshape(self.bias, (1, self.filters, 1))
                        outputs += bias
                    if self.rank == 2:
                        outputs = tf.nn.bias_add(outputs, self.bias, data_format="NCHW")
                    if self.rank == 3:
                        outputs_shape = outputs.shape.as_list()
                        outputs_4d = tf.reshape(outputs,
                                                [outputs_shape[0], outputs_shape[1],
                                                 outputs_shape[2] * outputs_shape[3],
                                                 otuputs_shape[4]])
                        outputs_4d = tf.nn.bias_add(outputs_4d, self.bias, data_format="NCHW")
                        outputs = tf.reshape(outputs_4d, outputs_shape)
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")
            # Activation
            if self.activation is not None:
                return self.activation(outputs)
            return outputs
    
        return tf.cond(training, conv_sp_var_drop, deterministic)

    def get_alpha(self):
        return tf.exp(self.log_sigma2 - tf.log(eps+self.kernel*self.kernel))

class DenseSpVarDrop(layers.Dense):
    """Sparse Variational Dropout for fully connected layer
    (Molchanov, 2015)
    """
    def __init__(self,
                 init_log_sigma2=-10,
                 log_sigma2_reg=None,
                 use_log_sigma2_bias=False,
                 name="SpVarDropDense",
                 **kwargs):
        super(ConvSpVarDrop, self).__init__(**kwargs)
        self.init_log_sigma2 = init_log_sigma2
        self.log_sigma2_reg = log_sigma2_reg
        self.use_log_sigma2_bias = use_log_sigma2_bias
        self.name = name

    def build(self, input_shape, dropout_mode="weights"):
        input_shape = tf.TensorShape(input_shape)
        self.log_sigma2 = self.add_variable("log_sigma2", 
                            shape=None,
                            initializer=tf.fill([input_shape[-1].value, self.units], 
                                                tf.log(self.init_log_sigma2)),
                            regularizer=self.log_sigma2_reg,
                            dtype=self.dtype,
                            trainable=True)
        if self.use_log_sigma2_bias:
            print("Have not implemented!."
        else:
            self.alpha_bias = None
        super(DenseSpVarDrop, self).build(input_shape)

    def get_alpha(self):
        return tf.exp(self.log_sigma2 - tf.log(eps+self.kernel*self.kernel))

    def call(self, inputs, training, thres=3, train_prune=False):
        def sp_vd_dropout():
            # Variational Dropout
            tinputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tinputs.get_shape().as_list()
            output_shape = shape[:1] + [self.units]
            kernel_sq = self.kernel * self.kernel
            # Local reparameterization trick
            noise = tf.random_normal(shape=output_shape)
            alpha = tf.clip_by_value(tf.exp(self.log_sigma2 - 2*tf.log(kern)), -8, 8)
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

def denseSpVD(
    inputs,
    training,
    init_alpha=1.0,
    alpha_reg=False,
    use_alpha_bias=False,
    name="VarDropDense",
    **kwargs):
    layer = DenseVarDrop(init_alpha, alpha_reg,
                         use_alpha_bias, **kwargs)
    return layer.apply(inputs, training=training), layer

def convSpVD(
    inputs,
    training,
    init_alpha=1.0,
    alpha_reg=False,
    use_alpha_bias=False,
    name="VarDropConv",
    **kwargs):
    layer = ConvVarDrop(init_alpha, alpha_reg,
                        use_alpha_bias, **kwargs)
    return layer.apply(inputs, training=training), layer

def vd_reg(alpha, constant=0.5):
    """Compute DK divergece between approximate
    posterior and the prior. This term is refered 
    as a regularization term prefering the posterior
    to be similar to the prior."""
    # TODO: Think about the constant value
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921
    return -tf.reduce_sum(constant + 0.5*tf.log(alpha) +\
                          tf.multiply(c1,alpha) +\
                          tf.multiply(c2,tf.pow(alpha,2)) +\
                          tf.multiply(c3,tf.pow(alpha,3)))

import os,sys
import tensorflow as tf


def conv2d(X, Out, Kernel, Stride, Padding='VALID', Name=None , act = None):

    padding = Kernel // 2
    padded_input = tf.pad(X, [[0,0],[padding,padding], [padding,padding], [0,0]], mode='REFLECT')
    with tf.variable_scope(Name):
        return tf.layers.conv2d(padded_input, Out, Kernel, Stride, padding=Padding, kernel_initializer=tf.random_normal_initializer(0.0, 0.01), bias_initializer=tf.constant_initializer(0.0))


def Wconv2d(X, W, strides=[1, 1, 1, 1], p='VALID', name=None):
    # set convolution layers.
    assert isinstance(X, tf.Tensor)
    print("W : ", W)
    if W[1] == 1 :
        padding = W[2] //2
    else:
        padding = W[1] // 2
    padded_input = tf.pad(X, [[0,0],[padding,padding], [padding,padding], [0,0]], mode='REFLECT')
    return tf.nn.conv2d(padded_input, W, strides=strides, padding=p, name=name)

def upconv2d(X, Out, Kernel, Stride, Padding='VALID', Name=None):
    with tf.variable_scope(Name):

        b, w, h, c = X.get_shape().as_list()

        X  = tf.image.resize_nearest_neighbor(X, [h * Stride, w*Stride])
      
        return conv2d(X, Out, Kernel, 1, Padding, Name=Name+'conv')    


def ReLU(X):
    return tf.nn.relu(X)

def Sigmoid(X):
    return tf.math.sigmoid(X)


def fc_layer(X, Out, Name=None):
    with tf.variable_scope(Name):
        return tf.layers.dense(X, Out)

def label_condition_variable(name, initializer, labels, num_categories, input_shape):
    shape = tf.TensorShape([num_categories]).concatenate(input_shape[-1:])
    var = tf.get_variable(name, shape, tf.float32, initializer=initializer, trainable=True)
    conditioned_var = tf.gather(var,labels)
    return tf.expand_dims(tf.expand_dims(conditioned_var,1),1)

def CIN(X, labels, num_categories, name=None):

    with tf.variable_scope(name):

        shape = X.get_shape()
        beta, gamma = None, None
        beta = label_condition_variable('beta', tf.zeros_initializer(), labels, num_categories, shape)
        gamma = label_condition_variable('gamma', tf.ones_initializer(), labels, num_categories, shape)

        mean, variance = tf.nn.moments(X, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        output = tf.nn.batch_normalization(X, mean, variance, beta, gamma, epsilon)
        output.set_shape(shape)
    
        return output
    


def Residual_block(X, Out, Kernel,Stride, labels, num_categories, Name=None):
    
    h = CIN(ReLU(conv2d(X, Out, Kernel, Stride, 'VALID', Name+'_conv_1')), labels, num_categories, name =Name+'residual_CIN_1')
    h = CIN(conv2d(h, Out, Kernel, Stride, 'VALID', Name+'_conv_2'), labels, num_categories, name=Name+'residual_CIN_2')
    
    return X + h


def gram_matrix(x):
    
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [-1, h*w, ch])
    # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
    return gram

def total_variation_loss(stylized_inputs):

    shape = tf.shape(stylized_inputs)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    y_size = tf.to_float((height - 1) * width * channels)
    x_size = tf.to_float(height * (width - 1) * channels)
    y_loss = tf.nn.l2_loss(stylized_inputs[:, 1:, :, :] - stylized_inputs[:, :-1, :, :]) / y_size
    x_loss = tf.nn.l2_loss(stylized_inputs[:, :, 1:, :] - stylized_inputs[:, :, :-1, :]) / x_size
    loss = (y_loss + x_loss) / tf.to_float(batch_size)

    return loss

def total_variation_regularization(X, beta=1):
    
    assert isinstance(X, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: Wconv2d(x, wh, p='VALID')
    tvw = lambda x: Wconv2d(x, ww, p='VALID')
    dh = tvh(X)
    dw = tvw(X)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    
    return tv

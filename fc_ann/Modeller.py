# coding=utf-8
import tensorflow as tf

'''
Created on 16.04.2018

@author: Arnold Riemer
'''
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    #initial = tf.truncated_normal(shape,dtype=tf.float64, stddev=0.1)
    #initial = tf.random_normal(shape,dtype=tf.float64, stddev=0.1)
    
    initial = tf.random_uniform(shape, -0.5, 0.5, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1,dtype=tf.float64, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_layer_withoutBias(inputs, in_size, out_size, activation_function=None, layer_name="Layer", maske=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = weight_variable(shape=[in_size, out_size])
            if not maske is None:
                Weights_masked = tf.multiply(Weights, maske)
                #Weights = tf.assign(Weights, tf.multiply(Weights, maske) )
                #Weights = tf.assign(Weights, tf.multiply(Weights.initialized_value(), tf.constant(maske)))
                
            else:
                Weights_masked = Weights
            variable_summaries(Weights)
        #with tf.name_scope('biases'):    
            #biases = bias_variable([1, out_size])
            #biases = bias_variable(shape=[out_size])
            #variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights_masked)
            tf.summary.histogram('h_layer', Wx_plus_b)
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('output', Wx_plus_b)
        biases = None    
        return outputs, Weights, biases, Weights_masked


def add_layer(inputs, in_size, out_size, activation_function=None, layer_name="Layer", maske=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = weight_variable(shape=[in_size, out_size])
            if not maske is None:
                Weights_masked = tf.multiply(Weights, maske)
                #Weights = tf.assign(Weights, tf.multiply(Weights, maske) )
                #Weights = tf.assign(Weights, tf.multiply(Weights.initialized_value(), tf.constant(maske)))
            else:
                Weights_masked = Weights
            variable_summaries(Weights)
        with tf.name_scope('biases'):    
            #biases = bias_variable([1, out_size])
            biases = bias_variable(shape=[out_size])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs, Weights_masked) , biases )
            tf.summary.histogram('h_layer', Wx_plus_b)
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('output', Wx_plus_b)
        return outputs, Weights, biases, Weights_masked

def add_layer_fixed(inputs, weight, bias, activation_function=None,layer_name="Layer", maske=None,  ):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = Weights = tf.Variable(weight)
            variable_summaries(Weights)
        with tf.name_scope('biases'):    
            biases = tf.Variable(bias)
            variable_summaries(biases)
        if not maske is None:
            Weights = tf.multiply(Weights, maske)       
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram('h_layer', Wx_plus_b)
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('output', Wx_plus_b)
        return outputs, Weights, biases

#============================================================================================================================================

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
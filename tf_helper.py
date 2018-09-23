import tensorflow as tf
import numpy as np

def tf_flatten(x, debug=False):
    flattened = tf.contrib.layers.flatten(x) 
    
    if debug:
        print("tf_flatten:")
        print("\tFlattened ", x.shape, " to ", flattened.shape)
        print("\n")
        
    return flattened

def tf_flatten_combine(x1, x2, debug=False):
    if debug:
        print("tf_flatten_combine: (showing internal functions)")
        print("\n")
    
    # flatten each input
    f1 = tf_flatten(x1, debug=debug)
    f2 = tf_flatten(x2, debug=debug)
    
    # combine each input together
    combined = tf.concat([f1, f2], axis=1) # tensor flow docs: https://www.tensorflow.org/api_docs/python/tf/concat
    
    if debug:
        print("\n")
        print("\ttf_flatten_combine output: ", combined.shape)
        print("\n")

    return combined
        
def conv2d(x, filter_dim, stride=1, w_mean=0, w_stddev=0.1, relu=False, debug=False):
    padding = 'VALID'
    
    input_dim = (x.shape[1].value, x.shape[2].value, x.shape[3].value)

    # Filter (weights and bias)
    F_W = tf.Variable(tf.truncated_normal((filter_dim[0], filter_dim[1], input_dim[2], filter_dim[2]), mean=w_mean, stddev=w_stddev))
    F_b = tf.Variable(tf.zeros(filter_dim[2]))
    strides = [1, stride, stride, 1]
    
    conv = tf.nn.conv2d(x, F_W, strides, padding) + F_b
    
    if debug:
        print("conv2d:")
        print("\tFilter: ", filter_dim[0], "x", filter_dim[1], "x", filter_dim[2])
        print("\tStride:  (" + str(stride) + ", " + str(stride) + ")")
        print("\tInput : ", x.shape)
        print("\tOutput: ", conv.shape)
        print("\n")
    
    # Apply ReLU Activation Function if parameter set to True (False by default)
    if relu:
        return tf.nn.relu(conv)
    else:
        return conv

def fullyConnectedLayer(x, n_output, w_mean=0, w_stddev=0.1, relu=False, debug=False):
    n_input = x.shape[1].value

    fc_W = tf.Variable(tf.truncated_normal(shape=(n_input, n_output), mean=w_mean, stddev=w_stddev))
    fc_b = tf.Variable(tf.zeros(n_output))
    
    fc = tf.matmul(x, fc_W) + fc_b

    if debug:
        print("fullyConnectedLayer:")
        print("\tInput : ", x.shape)
        print("\tOutput: ", fc.shape)
        print("\n")
    
    # Apply ReLU Activation Function if parameter set to True (False by default)
    if relu:
        return tf.nn.relu(fc)
    else:
        return fc
    
def maxpool2d(x, k=2, stride=1, debug=False):
    pooled = tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, stride, stride, 1],
        padding='VALID')
    
    if debug:
        print("maxpool2d:")
        print("\tKSize :  (" + str(k) + ", " + str(k) + ")")
        print("\tStride:  (" + str(stride) + ", " + str(stride) + ")")
        print("\tInput : ", x.shape)
        print("\tOutput: ", pooled.shape)
        print("\n")
        
    return pooled

def maxpool_dropout(x, drop_rate=0.5, k=2, stride=1, debug=False):
    out = maxpool2d(x=x, k=k, stride=stride, debug=debug)

    if debug:
        print("* applied dropout of ", drop_rate)
        print("\n")
    
    return tf.layers.dropout(out, rate=drop_rate)
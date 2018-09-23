import tensorflow as tf
import numpy as np
from tf_helper import *

# LeNet-5 architecture
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    debug = True
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = conv2d(x=x, filter_dim=(5, 5, 6), stride=1, w_mean=mu, w_stddev=sigma, debug=debug)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(x=conv1, filter_dim=(5, 5, 16), stride=1, w_mean=mu, w_stddev=sigma, debug=debug)
   
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = tf.contrib.layers.flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = fullyConnectedLayer(x=fc0, n_output=120, w_mean=mu, w_stddev=sigma)
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = fullyConnectedLayer(x=fc1, n_output=84, w_mean=mu, w_stddev=sigma)
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = fullyConnectedLayer(x=fc2, n_output=43, w_mean=mu, w_stddev=sigma)

    return logits
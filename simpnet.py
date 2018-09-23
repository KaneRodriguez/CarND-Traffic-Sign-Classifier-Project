import tensorflow as tf
import numpy as np
from tf_helper import *

# SimpNet (adapted from SimpNet Arch1 found in https://arxiv.org/pdf/1802.06205.pdf)
def SimpNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    debug = True
        
    # Define initial filter dimension
    fil_dim = (3, 3, 16) # H, W, K
 
    ''' Convolution Layer 1
    
        Note: relu=True signifies a ReLU activation function being applied internally
    '''
    conv = conv2d(x=x, filter_dim=fil_dim, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)

    ''' Convolution Layers 2, 3, and 4
    '''
    fil_dim=(3, 3, 44)
    
    for i in range(3):   
        conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
        
    ''' Convolution Layer 5
    '''
    fil_dim=(3, 3, 60)
    
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    ''' Max Pool and Dropout 1
    '''
    drop_rate = 0.5
    
    conv = maxpool_dropout(x=conv, drop_rate=drop_rate, k=2, stride=1)
    
    ''' Convolution Layers 6, 7, 8, and 9
    '''
    fil_dim=(3, 3, 60)
    
    for i in range(4):   
        conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    ''' Convolution Layer 10
    '''
    fil_dim=(3, 3, 128)
    
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    ''' Max Pool and Dropout 2
    '''
    drop_rate = 0.5
    
    conv = maxpool_dropout(x=conv, drop_rate=drop_rate, k=2, stride=1)
    
    ''' Convolution Layer 11
    '''
    fil_dim=(3, 3, 128)
    
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    ''' Convolution Layer 12
    '''
    fil_dim=(3, 3, 180)
    
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    ''' Convolution Layer 13
    '''
    fil_dim=(3, 3, 230)
    
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)

    ''' Global Max Pool 1
    '''
    width = conv.shape[2].value
    height = conv.shape[1].value
    
    # reduce tensor from dimensions (w x h x d) to (1 x 1 x d)
    conv = tf.layers.max_pooling2d(conv, pool_size=[height, width], strides=1)
    
    # Flatten tensor from (1 x 1 x d) to d
    fc   = tf.contrib.layers.flatten(conv) 
    
    ''' Fully Connected Layer 1
    '''
    d = fc.shape[1].value
    n_classifiers = 43
    
    # Fully connect the input from the last layer (d) to the number of desired classifiers
    fc = fullyConnectedLayer(x=fc, n_input=d, n_output=n_classifiers, w_mean=mu, w_stddev=sigma)
    
    # Activation.
    logits    = tf.nn.relu(fc)
    
    return logits
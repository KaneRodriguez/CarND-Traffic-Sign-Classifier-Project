import tensorflow as tf
import numpy as np
from tf_helper import *

# Modified LeNet-5 architecture
def ModLeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    debug = True
    
    # Define initial filter dimension
    fil_dim = (5, 5, 6) # H, W, K
 
    # print("Convolutional Layer 1...\n")
    # Note: relu=True signifies a ReLU activation function being applied internally
    conv = conv2d(x=x, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)
    
    # print("Max Pool and Dropout 1...\n")
    drop_rate = 0.0
    
    conv = maxpool_dropout(x=conv, drop_rate=drop_rate, k=2, stride=2, debug=debug)
 
    # print("Convolutional Layer 2...\n")
    fil_dim = (5, 5, 16)
    conv = conv2d(x=conv, filter_dim=fil_dim, stride=1, w_mean=mu, w_stddev=sigma, relu=True, debug=debug)

    # print("Max Pool and Dropout 2...\n")
    drop_rate = 0.0
    
    conv = maxpool_dropout(x=conv, drop_rate=drop_rate, k=2, stride=2, debug=debug)

    # Flatten tensor from (h x w x d) to the result of h*w*d
    fc   = tf_flatten(x=conv, debug=debug)
    
    # print("Fully Connected Layer 1...\n")
    n_classifiers = 120
    
    # Fully connect the input from the last layer to the number of desired classifiers
    fc = fullyConnectedLayer(x=fc, n_output=n_classifiers, relu=True, w_mean=mu, w_stddev=sigma, debug=debug)
    
    # print("Fully Connected Layer 2...\n")
    n_classifiers = 84
    
    fc = fullyConnectedLayer(x=fc, n_output=n_classifiers, relu=True, w_mean=mu, w_stddev=sigma, debug=debug)
       
    # print("Fully Connected Layer 3...\n")
    n_classifiers = 43
    
    logits = fullyConnectedLayer(x=fc, n_output=n_classifiers, w_mean=mu, w_stddev=sigma, debug=debug)
       
    return logits
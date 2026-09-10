import tensorflow as tf
import numpy as np

def load_mnist_data(num_train=1000, num_test=500):
    """
    Loads MNIST dataset and returns preprocessed training and testing sets.
    
    Args:
        num_train (int): Number of training samples to return.
        num_test (int): Number of testing samples to return.
        
    Returns:
        tuple: (X_train, Y_train), (X_test, Y_test)
    """
    print(f"Loading MNIST data (Train: {num_train}, Test: {num_test})...")
    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    
    

    
    if num_train > 0:
        X_train = x_train[:num_train].reshape(num_train, -1) / 255.0
        Y_train = y_train[:num_train]
    else:
        X_train = np.empty((0, 784))
        Y_train = np.empty((0,))
    
    if num_test > 0:
        X_test = x_test[:num_test].reshape(num_test, -1) / 255.0
        Y_test = y_test[:num_test]
    else:
        X_test = np.empty((0, 784))
        Y_test = np.empty((0,))
    
    return (X_train, Y_train), (X_test, Y_test)

def get_raw_mnist():
    """Returns the raw MNIST dataset."""
    return tf.keras.datasets.mnist.load_data()

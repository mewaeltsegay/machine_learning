import cupy as cp
from tensorflow.keras.datasets import mnist

def load_mnist():
    """
    Load MNIST dataset using Keras and normalize to [-1, 1]
    """
    # Load MNIST
    (X_train, _), (X_test, _) = mnist.load_data()
    
    # Reshape and add channel dimension
    X_train = X_train.reshape(-1, 1, 28, 28).astype('float32')
    X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')
    
    # Normalize to [-1, 1] and convert to CuPy arrays
    X_train = cp.array((X_train / 127.5) - 1)
    X_test = cp.array((X_test / 127.5) - 1)
    
    return X_train, X_test

def get_batches(data, batch_size):
    """
    Generate batches from data
    """
    num_samples = len(data)
    num_batches = num_samples // batch_size
    
    # Shuffle data using CuPy
    indices = cp.random.permutation(num_samples)
    data = data[indices]
    
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]
import numpy as np
from keras.datasets import mnist

def load_mnist():
    # Load MNIST dataset using Keras
    (x_train, _), (_, _) = mnist.load_data()
    
    # Normalize to [0,1] and reshape to (n_samples, 784)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    
    return x_train

def get_batches(data, batch_size):
    # Shuffle the data
    indices = np.random.permutation(len(data))
    data = data[indices]
    
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size] 
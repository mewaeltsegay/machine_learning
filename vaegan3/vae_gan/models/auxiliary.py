import numpy as np
from .layers import Dense, Activation, relu, relu_prime, sigmoid, sigmoid_prime

class Auxiliary:
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        self.decoder_layers = [
            Dense(latent_dim + 10, hidden_dim),  # +10 for label
            Activation(relu, relu_prime),
            Dense(hidden_dim, output_dim),
            Activation(sigmoid, sigmoid_prime)
        ]
        
    def decode(self, z, y, label=None, alpha=1.0):
        # Convert y to one-hot
        if label is not None:
            y_onehot = np.zeros((y.shape[0], 10))
            y_onehot[np.arange(y.shape[0]), y] = alpha
            if label is not None:
                y_onehot[np.arange(y.shape[0]), label] = 1-alpha
        else:
            y_onehot = np.zeros((y.shape[0], 10))
            y_onehot[np.arange(y.shape[0]), y] = 1
            
        # Concatenate z with one-hot label
        h = np.concatenate([z.reshape(z.shape[0], -1), y_onehot], axis=1)
        
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h
        
    def forward(self, z, y, label=None, alpha=1.0):
        return self.decode(z, y, label, alpha)
        
    def backward(self, gradient, optimizer):
        """Backward pass through decoder layers"""
        for layer in reversed(self.decoder_layers):
            gradient = layer.backward(gradient, optimizer)
        return gradient
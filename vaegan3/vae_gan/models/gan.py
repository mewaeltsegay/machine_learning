import numpy as np
from .layers import Dense, Activation, sigmoid, sigmoid_prime, relu, relu_prime

def idx2onehot(idx, n_classes):
    """Convert index to one-hot vector"""
    onehot = np.zeros((idx.shape[0], n_classes))
    onehot[np.arange(idx.shape[0]), idx] = 1
    return onehot

class Discriminator:
    def __init__(self, input_dim=784):
        # Label embedding
        self.label_embedding = Dense(10, 10)
        
        self.layers = [
            Dense(input_dim + 10, 1024),  # +10 for embedded label
            Activation(relu, relu_prime),
            Dense(1024, 512),
            Activation(relu, relu_prime), 
            Dense(512, 256),
            Activation(relu, relu_prime),
            Dense(256, 1),
            Activation(sigmoid, sigmoid_prime)
        ]
    
    def forward(self, x, y):
        # Convert labels to one-hot and embed
        y_onehot = idx2onehot(y, 10)
        y_emb = self.label_embedding.forward(y_onehot)
        
        # Reshape input and concatenate with embedded label
        h = np.concatenate([x.reshape(x.shape[0], -1), y_emb], axis=1)
        
        features = None
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i == 4:  # Store features after 256-dim layer
                features = h
                
        return features, h
    
    def compute_gradient(self, output_grad):
        # Backpropagate to get input gradients
        grad = output_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, None)  # None as we don't update weights here
        return grad
    
    def backward(self, gradient, optimizer):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, optimizer)
            
class Generator:
    def __init__(self, latent_dim=20, hidden_dim=256, output_dim=784):
        self.layers = [
            Dense(latent_dim, hidden_dim),
            Activation(relu, relu_prime),
            Dense(hidden_dim, hidden_dim),
            Activation(relu, relu_prime),
            Dense(hidden_dim, output_dim),
            Activation(sigmoid, sigmoid_prime)
        ]
    
    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer.forward(h)
        return h
    
    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate) 
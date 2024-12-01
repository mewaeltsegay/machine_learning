import numpy as np
from .layers import Dense, Activation, sigmoid, sigmoid_prime, relu, relu_prime


class VAE:
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.training = True
        
        # Encoder layers
        self.encoder_layers = [
            Dense(input_dim + 10, hidden_dim),  # +10 for one-hot label
            Activation(relu, relu_prime),
            Dense(hidden_dim, latent_dim),  # For mean
            Dense(hidden_dim, latent_dim)   # For logvar
        ]
        
        # Remove complex conv layers and replace with simple linear layers
        self.mean_layer = self.encoder_layers[2]
        self.logvar_layer = self.encoder_layers[3]
    
    def train(self):
        """Set model to training mode"""
        self.training = True
        
    def eval(self):
        """Set model to evaluation mode"""
        self.training = False
    
    def encode(self, x, y, label=None, alpha=1.0):
        # Convert y to one-hot and concatenate with x
        if label is not None:
            y_onehot = np.zeros((y.shape[0], 10))
            y_onehot[np.arange(y.shape[0]), y] = alpha
            if label is not None:
                y_onehot[np.arange(y.shape[0]), label] = 1-alpha
        else:
            y_onehot = np.zeros((y.shape[0], 10))
            y_onehot[np.arange(y.shape[0]), y] = 1
            
        h = np.concatenate([x, y_onehot], axis=1)
        
        # Forward through encoder layers
        h = self.encoder_layers[0].forward(h)
        h = self.encoder_layers[1].forward(h)
        
        mean = self.mean_layer.forward(h)
        logvar = self.logvar_layer.forward(h)
        
        return mean, logvar
        
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        if self.training:
            eps = np.random.standard_normal(mu.shape)
            return mu + eps * std
        return mu
            
    def forward(self, x, y, label=None, alpha=1.0):
        mu, logvar = self.encode(x, y, label, alpha)
        z = self.reparameterize(mu, logvar)
        return mu, logvar
        
    def backward(self, gradient, optimizer):
        """Backward pass through encoder layers"""
        # Split gradient for mean and logvar
        batch_size = gradient.shape[0]
        grad_mu = gradient[:, :self.latent_dim]
        grad_logvar = gradient[:, self.latent_dim:]
        
        # Backward through mean and logvar layers separately
        grad_hidden_mu = self.mean_layer.backward(grad_mu, optimizer)
        grad_hidden_logvar = self.logvar_layer.backward(grad_logvar, optimizer)
        
        # Combine gradients
        grad_hidden = grad_hidden_mu + grad_hidden_logvar
        
        # Continue backward through remaining layers
        grad = self.encoder_layers[1].backward(grad_hidden, optimizer)
        grad = self.encoder_layers[0].backward(grad, optimizer)
        return grad
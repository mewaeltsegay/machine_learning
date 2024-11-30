import numpy as np

class Discriminator:
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Reference architecture: 784 -> 256 -> 64 -> 1
        input_dim = np.prod(self.input_shape)  # 784
        
        self.W1 = np.random.randn(input_dim, 256) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(256)
        
        self.W2 = np.random.randn(256, 64) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros(64)
        
        self.W3 = np.random.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(1)
    
    def forward(self, X, training=True):
        # Flatten input
        X = X.reshape(X.shape[0], -1)  # -> (batch, 784)
        
        # First layer + ReLU
        self.fc1 = np.dot(X, self.W1) + self.b1
        self.features_1 = np.maximum(0, self.fc1)  # Store features for perceptual loss
        
        # Second layer + ReLU
        self.fc2 = np.dot(self.features_1, self.W2) + self.b2
        self.features_2 = np.maximum(0, self.fc2)  # Store features for perceptual loss
        
        # Final layer + Sigmoid
        self.logits = np.dot(self.features_2, self.W3) + self.b3
        self.output = 1 / (1 + np.exp(-self.logits))
        
        return self.output, self.features_1, self.features_2
    
    def parameters(self):
        """Return a dictionary of all trainable parameters"""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3
        }
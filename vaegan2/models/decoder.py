import numpy as np

class Decoder:
    def __init__(self, latent_dim=2, output_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Mirror of encoder: 2 -> 64 -> 128 -> 512 -> 784
        self.W1 = np.random.randn(self.latent_dim, 64) * np.sqrt(2.0 / self.latent_dim)
        self.b1 = np.zeros(64)
        
        self.W2 = np.random.randn(64, 128) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(128)
        
        self.W3 = np.random.randn(128, 512) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(512)
        
        self.W4 = np.random.randn(512, np.prod(self.output_shape)) * np.sqrt(2.0 / 512)
        self.b4 = np.zeros(np.prod(self.output_shape))
    
    def forward(self, z, training=True):
        # Dense layers with ReLU
        self.fc1 = np.dot(z, self.W1) + self.b1
        self.fc1_act = np.maximum(0, self.fc1)  # ReLU
        
        self.fc2 = np.dot(self.fc1_act, self.W2) + self.b2
        self.fc2_act = np.maximum(0, self.fc2)  # ReLU
        
        self.fc3 = np.dot(self.fc2_act, self.W3) + self.b3
        self.fc3_act = np.maximum(0, self.fc3)  # ReLU
        
        # Final layer with sigmoid
        self.fc4 = np.dot(self.fc3_act, self.W4) + self.b4
        self.output = 1 / (1 + np.exp(-self.fc4))  # Sigmoid
        
        # Reshape to image dimensions
        return self.output.reshape(-1, *self.output_shape)
    
    def parameters(self):
        """Return a dictionary of all trainable parameters"""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
            'W4': self.W4,
            'b4': self.b4
        }
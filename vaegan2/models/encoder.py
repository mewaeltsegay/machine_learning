import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=(0, 1, 2))
            var = np.var(x, axis=(0, 1, 2))
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        x_norm = (x - mean.reshape(1, 1, 1, -1)) / np.sqrt(var.reshape(1, 1, 1, -1) + self.eps)
        
        # Scale and shift
        return self.gamma.reshape(1, 1, 1, -1) * x_norm + self.beta.reshape(1, 1, 1, -1)

class Encoder:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=2):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Following reference architecture: 784 -> 512 -> 128 -> 64 -> 2
        flattened_input = np.prod(self.input_shape)  # 784
        
        # Dense layers instead of conv (like reference)
        self.W1 = np.random.randn(flattened_input, 512) * np.sqrt(2.0 / flattened_input)
        self.b1 = np.zeros(512)
        
        self.W2 = np.random.randn(512, 128) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(128)
        
        self.W3 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(64)
        
        # Mean and logvar heads
        self.W_mean = np.random.randn(64, self.latent_dim) * np.sqrt(2.0 / 64)
        self.b_mean = np.zeros(self.latent_dim)
        
        self.W_logvar = np.random.randn(64, self.latent_dim) * np.sqrt(2.0 / 64)
        self.b_logvar = np.zeros(self.latent_dim)

    def forward(self, X, training=True):
        # Flatten input
        self.flattened = X.reshape(X.shape[0], -1)  # -> (batch, 784)
        
        # Dense layers with ReLU
        self.fc1 = np.dot(self.flattened, self.W1) + self.b1
        self.fc1_act = np.maximum(0, self.fc1)  # ReLU
        
        self.fc2 = np.dot(self.fc1_act, self.W2) + self.b2
        self.fc2_act = np.maximum(0, self.fc2)  # ReLU
        
        self.fc3 = np.dot(self.fc2_act, self.W3) + self.b3
        self.fc3_act = np.maximum(0, self.fc3)  # ReLU
        
        # Mean and log variance
        self.mean = np.dot(self.fc3_act, self.W_mean) + self.b_mean
        self.log_var = np.dot(self.fc3_act, self.W_logvar) + self.b_logvar
        
        # Reparameterization trick
        if training:
            epsilon = np.random.randn(X.shape[0], self.latent_dim)
        else:
            epsilon = 0
            
        self.z = self.mean + np.exp(0.5 * self.log_var) * epsilon
        
        return self.z, self.mean, self.log_var

    def parameters(self):
        """Return a dictionary of all trainable parameters"""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'gamma1': self.bn1.gamma,
            'beta1': self.bn1.beta,
            'W2': self.W2,
            'b2': self.b2,
            'gamma2': self.bn2.gamma,
            'beta2': self.bn2.beta,
            'W_mean': self.W_mean,
            'b_mean': self.b_mean,
            'W_logvar': self.W_logvar,
            'b_logvar': self.b_logvar
        }
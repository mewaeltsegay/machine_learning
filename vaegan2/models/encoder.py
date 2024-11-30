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
    def __init__(self, input_shape=(28, 28, 1), latent_dim=32):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Conv1: (28,28,1) -> (14,14,16)
        self.W1 = np.random.randn(3, 3, 1, 16) * np.sqrt(2.0 / (3 * 3 * 1))
        self.b1 = np.zeros((16,))
        self.bn1 = BatchNorm(16)
        
        # Conv2: (14,14,16) -> (7,7,32)
        self.W2 = np.random.randn(3, 3, 16, 32) * np.sqrt(2.0 / (3 * 3 * 16))
        self.b2 = np.zeros((32,))
        self.bn2 = BatchNorm(32)
        
        # Dense layers for mean and log_var
        flattened_dim = 7 * 7 * 32
        self.W_mean = np.random.randn(flattened_dim, self.latent_dim) * np.sqrt(1.0 / flattened_dim)
        self.b_mean = np.zeros((self.latent_dim,))
        
        self.W_logvar = np.random.randn(flattened_dim, self.latent_dim) * np.sqrt(1.0 / flattened_dim)
        self.b_logvar = np.zeros((self.latent_dim,))

    def conv2d(self, X, W, b, stride=2, padding=1):
        # Using im2col for faster convolution
        n, h, w, c = X.shape
        kh, kw, _, out_c = W.shape
        
        # Pad input
        X_pad = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
        
        out_h = (h + 2*padding - kh)//stride + 1
        out_w = (w + 2*padding - kw)//stride + 1
        
        # Im2col transformation
        X_col = self.im2col(X_pad, kh, kw, stride)
        W_col = W.reshape(out_c, -1).T
        
        output = np.dot(X_col, W_col) + b
        output = output.reshape(n, out_h, out_w, out_c)
        
        return output
    
    def im2col(self, X, kh, kw, stride):
        n, h, w, c = X.shape
        out_h = (h - kh)//stride + 1
        out_w = (w - kw)//stride + 1
        
        # Initialize output matrix
        col = np.zeros((n, kh*kw*c, out_h*out_w))
        
        # Fill the output matrix
        for y in range(out_h):
            y_start = y * stride
            for x in range(out_w):
                x_start = x * stride
                patch = X[:, y_start:y_start+kh, x_start:x_start+kw, :]
                col[:, :, y*out_w + x] = patch.reshape(n, -1)
        
        # Reshape to final form
        col = col.transpose(0, 2, 1)  # (n, out_h*out_w, kh*kw*c)
        col = col.reshape(-1, kh*kw*c)  # (n*out_h*out_w, kh*kw*c)
        
        return col
    
    def leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def forward(self, X, training=True):
        self.X = X
        
        # Conv1 + BatchNorm + LeakyReLU
        self.conv1 = self.conv2d(X, self.W1, self.b1)
        self.bn1_out = self.bn1.forward(self.conv1, training)
        self.conv1_act = self.leaky_relu(self.bn1_out, alpha=0.2)
        
        # Conv2 + BatchNorm + LeakyReLU
        self.conv2 = self.conv2d(self.conv1_act, self.W2, self.b2)
        self.bn2_out = self.bn2.forward(self.conv2, training)
        self.conv2_act = self.leaky_relu(self.bn2_out, alpha=0.2)
        
        # Flatten
        self.flattened = self.conv2_act.reshape(X.shape[0], -1)
        
        # Mean and log variance
        self.mean = np.dot(self.flattened, self.W_mean) + self.b_mean
        self.log_var = np.dot(self.flattened, self.W_logvar) + self.b_logvar
        
        # Reparameterization trick with reduced noise for MNIST
        if training:
            epsilon = np.random.randn(X.shape[0], self.latent_dim) * 0.1
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
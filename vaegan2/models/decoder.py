import numpy as np

class Decoder:
    def __init__(self, latent_dim=32, output_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Dense: latent_dim -> 7*7*32
        self.W1 = np.random.randn(self.latent_dim, 7 * 7 * 32) * 0.02
        self.b1 = np.zeros((7 * 7 * 32,))
        
        # Deconv1: (7,7,32) -> (14,14,16)
        self.W2 = np.random.randn(3, 3, 16, 32) * 0.02
        self.b2 = np.zeros((16,))
        
        # Deconv2: (14,14,16) -> (28,28,1)
        self.W3 = np.random.randn(3, 3, 1, 16) * 0.02
        self.b3 = np.zeros((1,))
        
    def deconv2d(self, X, W, b, output_shape, stride=2, padding=1):
        n, h, w, c = X.shape
        kh, kw, out_c, in_c = W.shape
        
        if output_shape[0] is None:
            output_shape = (n,) + output_shape[1:]
            
        out_h, out_w = output_shape[1:3]
        
        # Initialize output without padding (we'll add it later)
        output = np.zeros((n, out_h, out_w, out_c))
        
        # Add padding to output if needed
        if padding > 0:
            output = np.pad(output, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
        
        # Calculate output dimensions with padding
        padded_h = out_h + 2 * padding
        padded_w = out_w + 2 * padding
        
        # Perform deconvolution
        for i in range(h):
            for j in range(w):
                h_start = i * stride
                w_start = j * stride
                
                # Get input values for all channels
                input_val = X[:, i, j, :]  # Shape: (n, in_c)
                
                # Process each output channel
                for out_ch in range(out_c):
                    # Get weights for this output channel
                    weight = W[:, :, out_ch, :].reshape(kh * kw, in_c)  # Shape: (kh*kw, in_c)
                    
                    # Compute contribution
                    contrib = np.dot(input_val, weight.T)  # Shape: (n, kh*kw)
                    contrib = contrib.reshape(n, kh, kw)
                    
                    # Calculate valid output region
                    h_end = min(h_start + kh, padded_h)
                    w_end = min(w_start + kw, padded_w)
                    
                    # Calculate valid contribution region
                    ch = min(kh, padded_h - h_start)
                    cw = min(kw, padded_w - w_start)
                    
                    if ch > 0 and cw > 0:  # Only add if there's a valid region
                        output[:, h_start:h_end, w_start:w_end, out_ch] += contrib[:, :ch, :cw]
        
        # Add bias
        output = output + b.reshape(1, 1, 1, -1)
        
        # Remove padding if needed
        if padding > 0:
            output = output[:, padding:-padding, padding:-padding, :]
            
        return output
    
    def relu(self, X):
        return np.maximum(0, X)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def forward(self, z, training=True):
        # Dense + ReLU
        self.dense1 = np.dot(z, self.W1) + self.b1
        self.dense1_relu = self.relu(self.dense1)
        
        # Reshape
        self.reshaped = self.dense1_relu.reshape(-1, 7, 7, 32)
        
        # Deconv1 + ReLU
        self.deconv1 = self.deconv2d(self.reshaped, self.W2, self.b2, (None, 14, 14, 16))
        self.deconv1_relu = self.relu(self.deconv1)
        
        # Deconv2 + Sigmoid (better for MNIST binary images)
        batch_size = z.shape[0]
        self.deconv2 = self.deconv2d(self.deconv1_relu, self.W3, self.b3, (batch_size, 28, 28, 1))
        self.output = self.sigmoid(self.deconv2)
        
        return self.output
    
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
import numpy as np

class Discriminator:
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        # First initialize conv layers to get actual dimensions
        self.init_conv_layers()
        # Then initialize dense layer with correct dimensions
        self.init_dense_layer()
        
    def init_conv_layers(self):
        # Conv1: (28,28,1) -> (14,14,32)
        self.W1 = np.random.randn(3, 3, 1, 32) * np.sqrt(2.0 / (3 * 3 * 1))
        self.b1 = np.zeros((32,))
        
        # Conv2: (14,14,32) -> (7,7,64)
        self.W2 = np.random.randn(3, 3, 32, 64) * np.sqrt(2.0 / (3 * 3 * 32))
        self.b2 = np.zeros((64,))
        
        # Calculate actual flattened size using a dummy forward pass
        dummy_input = np.zeros((1, *self.input_shape))
        conv1 = self.conv2d(dummy_input, self.W1, self.b1)
        conv2 = self.conv2d(conv1, self.W2, self.b2)
        self.flattened_size = np.prod(conv2.shape[1:])
        
    def init_dense_layer(self):
        # Dense: (flattened_size) -> 1
        self.W3 = np.random.randn(self.flattened_size, 1) * np.sqrt(2.0 / self.flattened_size)
        self.b3 = np.zeros((1,))
        
    def conv2d(self, X, W, b, stride=2, padding=1):
        n, h, w, c = X.shape
        kh, kw, _, out_c = W.shape
        
        # Pad input
        X_pad = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
        
        out_h = (h + 2*padding - kh)//stride + 1
        out_w = (w + 2*padding - kw)//stride + 1
        
        output = np.zeros((n, out_h, out_w, out_c))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kh
                w_start = j * stride
                w_end = w_start + kw
                
                X_slice = X_pad[:, h_start:h_end, w_start:w_end, :]
                for k in range(out_c):
                    output[:, i, j, k] = np.sum(X_slice * W[:, :, :, k], axis=(1,2,3)) + b[k]
        
        return output
    
    def leaky_relu(self, X, alpha=0.01):
        return np.where(X > 0, X, alpha * X)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def forward(self, X, training=True):
        # If input is not 4D, try to reshape it
        if len(X.shape) != 4:
            try:
                X = X.reshape(X.shape[0], 28, 28, 1)
            except ValueError as e:
                raise ValueError(f"Failed to reshape input from {X.shape} to (batch, 28, 28, 1)")
        
        # Ensure input has correct shape
        if X.shape[1:] != self.input_shape:
            total_elements = np.prod(X.shape)
            expected_elements = X.shape[0] * np.prod(self.input_shape)
            
            raise ValueError(
                f"Input shape {X.shape} is invalid. "
                f"Expected shape (batch, 28, 28, 1), "
                f"got {X.shape}. "
                f"Total elements: {total_elements}, "
                f"Expected: {expected_elements}"
            )
        
        # Conv1 + LeakyReLU: (28,28,1) -> (14,14,32)
        self.conv1 = self.conv2d(X, self.W1, self.b1)
        self.conv1_act = self.leaky_relu(self.conv1)
        
        # Conv2 + LeakyReLU: (14,14,32) -> (7,7,64)
        self.conv2 = self.conv2d(self.conv1_act, self.W2, self.b2)
        self.conv2_act = self.leaky_relu(self.conv2)
        
        # Flatten
        self.flattened = self.conv2_act.reshape(X.shape[0], -1)
        
        # Dense + Sigmoid
        self.logits = np.dot(self.flattened, self.W3) + self.b3
        self.output = self.sigmoid(self.logits)
        
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
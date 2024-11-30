import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                      np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros((out_channels, 1))
        
        # Gradients
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)
        
        # Cache for backward pass
        self.cache = None
        
    def forward(self, X):
        n_samples, c, h, w = X.shape
        
        if self.padding > 0:
            X_padded = np.pad(X, 
                ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 
                mode='constant')
        else:
            X_padded = X
        
        # Calculate output dimensions
        h_out = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Ensure output dimensions are at least 1
        h_out = max(1, h_out)
        w_out = max(1, w_out)
        
        output = np.zeros((n_samples, self.out_channels, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # Ensure we don't exceed input dimensions
                h_end = min(h_end, X_padded.shape[2])
                w_end = min(w_end, X_padded.shape[3])
                
                if h_end > h_start and w_end > w_start:
                    X_patch = X_padded[:, :, h_start:h_end, w_start:w_end]
                    
                    # Handle incomplete patches
                    if X_patch.shape[2:] != (self.kernel_size, self.kernel_size):
                        pad_h = self.kernel_size - X_patch.shape[2]
                        pad_w = self.kernel_size - X_patch.shape[3]
                        X_patch = np.pad(X_patch, 
                            ((0,0), (0,0), (0,pad_h), (0,pad_w)), 
                            mode='constant')
                    
                    for k in range(self.out_channels):
                        output[:, k, i, j] = np.sum(X_patch * self.weights[k, :, :, :], 
                                                  axis=(1,2,3)) + self.biases[k]
        
        self.cache = (X, X_padded)
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def backward(self, dout):
        X, X_padded = self.cache
        n_samples, _, h_out, w_out = dout.shape
        
        if self.activation is not None:
            dout = dout * self.activation.derivative(dout)
        
        # Initialize gradients
        dX_padded = np.zeros_like(X_padded)
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)
        
        # Compute gradients
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                for k in range(self.out_channels):
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k, :, :, :] * dout[:, k, i, j][:, None, None, None]
                    self.d_weights[k] += np.sum(
                        X_padded[:, :, h_start:h_end, w_start:w_end] * \
                        dout[:, k, i, j][:, None, None, None],
                        axis=0
                    )
                    self.d_biases[k] += np.sum(dout[:, k, i, j])
        
        # Remove padding from dX if necessary
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        return dX

    def get_params(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_params(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

    def get_gradients(self):
        return {
            'weights': self.d_weights,
            'biases': self.d_biases
        }

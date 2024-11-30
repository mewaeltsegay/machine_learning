import numpy as np

class TransposedConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.activation = activation
        
        # He initialization
        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        self.weights = np.random.normal(0, scale, 
            (in_channels, out_channels, *self.kernel_size))
        self.biases = np.zeros(out_channels)
        
        # Gradient placeholders
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.biases)
        self.last_input = None

    def _create_output_patches(self, batch_size, h_out, w_out):
        """Create matrix of output patch positions"""
        h_in = (h_out + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_in = (w_out + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        patches = np.zeros((batch_size, self.out_channels, h_in * w_in, 
                          self.kernel_size[0] * self.kernel_size[1]))
        
        for i in range(h_in):
            for j in range(w_in):
                h_start = i * self.stride[0]
                w_start = j * self.stride[1]
                idx = i * w_in + j
                patches[:, :, idx] = np.arange(
                    h_start * w_out + w_start,
                    (h_start + self.kernel_size[0]) * w_out + w_start + self.kernel_size[1]
                ).reshape(-1)
        return patches.astype(int)

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"Input must be 4D, got {x.shape}")
        
        self.last_input = x
        batch_size, in_channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        w_out = (w_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        
        # Reshape weights to (in_channels, out_channels * kernel_size[0] * kernel_size[1])
        w_reshaped = self.weights.reshape(
            self.in_channels, 
            self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        
        # Reshape input to (batch_size, in_channels, h_in * w_in)
        x_reshaped = x.reshape(batch_size, in_channels, h_in * w_in)
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Compute transposed convolution for each batch
        for b in range(batch_size):
            # Perform matrix multiplication
            temp = np.dot(w_reshaped.T, x_reshaped[b])  # Shape: (out_ch * kh * kw, h_in * w_in)
            
            # Reshape the result
            temp = temp.reshape(self.out_channels, self.kernel_size[0], 
                              self.kernel_size[1], h_in * w_in)
            
            # Initialize the output for this batch
            out_b = np.zeros((self.out_channels, h_out, w_out))
            
            # Distribute values to output
            for i in range(h_in):
                for j in range(w_in):
                    h_start = i * self.stride[0]
                    w_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[0]
                    w_end = w_start + self.kernel_size[1]
                    
                    if h_end <= h_out and w_end <= w_out:
                        out_b[:, h_start:h_end, w_start:w_end] += \
                            temp[:, :, :, i * w_in + j]
            
            output[b] = out_b
        
        # Add biases
        output += self.biases.reshape(1, -1, 1, 1)
        
        self.last_output = output
        
        # Apply activation
        return self.activation(output) if self.activation else output

    def backward(self, output_grad):
        if self.last_input is None or self.last_output is None:
            raise ValueError("No cached input/output found. Did you call forward first?")
        
        batch_size = self.last_input.shape[0]
        
        # Apply activation gradient if necessary
        if self.activation is not None:
            output_grad = output_grad * self.activation.derivative(self.last_output)
        
        # Get shapes
        _, _, h_in, w_in = self.last_input.shape
        _, _, h_out, w_out = output_grad.shape
        
        # Compute bias gradients
        self.bias_grad = np.sum(output_grad, axis=(0, 2, 3))
        
        # Initialize gradients
        self.weight_grad = np.zeros_like(self.weights)
        input_grad = np.zeros_like(self.last_input)
        
        # Compute weight gradients
        for b in range(batch_size):
            x_b = self.last_input[b]  # (in_channels, h_in, w_in)
            grad_b = output_grad[b]   # (out_channels, h_out, w_out)
            
            # For each spatial position in the input
            for i in range(h_in):
                for j in range(w_in):
                    # Compute corresponding region in gradient
                    h_start = i * self.stride[0]
                    w_start = j * self.stride[1]
                    h_end = min(h_start + self.kernel_size[0], h_out)
                    w_end = min(w_start + self.kernel_size[1], w_out)
                    
                    if h_start < h_out and w_start < w_out:
                        # Extract patches
                        grad_patch = grad_b[:, h_start:h_end, w_start:w_end]  # (out_channels, kh, kw)
                        input_val = x_b[:, i, j]  # (in_channels,)
                        
                        # Update weights using outer product
                        for ic in range(self.in_channels):
                            for oc in range(self.out_channels):
                                self.weight_grad[ic, oc, :h_end-h_start, :w_end-w_start] += (
                                    input_val[ic] * grad_patch[oc]
                                )
        
        # Compute input gradients
        padded_grad = np.pad(
            output_grad,
            ((0, 0), (0, 0), 
             (self.kernel_size[0]-1, self.kernel_size[0]-1),
             (self.kernel_size[1]-1, self.kernel_size[1]-1)),
            mode='constant'
        )
        
        # Flip weights for convolution
        flipped_weights = np.flip(np.flip(self.weights, axis=2), axis=3)
        
        # Compute input gradients
        for b in range(batch_size):
            for i in range(h_in):
                for j in range(w_in):
                    h_start = i * self.stride[0]
                    w_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[0]
                    w_end = w_start + self.kernel_size[1]
                    
                    grad_patch = padded_grad[b, :, h_start:h_end, w_start:w_end]
                    
                    for ic in range(self.in_channels):
                        for oc in range(self.out_channels):
                            input_grad[b, ic, i, j] += np.sum(
                                flipped_weights[ic, oc] * grad_patch[oc]
                            )
        
        # Normalize gradients by batch size
        self.weight_grad /= batch_size
        self.bias_grad /= batch_size
        
        return input_grad

    def get_params(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_params(self, params):
        self.weights = params['weights']
        self.biases = params['biases']


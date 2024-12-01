import numpy as np

# First define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=1.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, w, grad_w):
        grad_w = np.clip(grad_w, -self.clip_value, self.clip_value)
        
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_w)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, optimizer):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=relu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((out_channels,))
        
        self.activation = activation
        self.activation_prime = relu_prime if activation == relu else None
        
        self.weight_optimizer = Adam()
        self.bias_optimizer = Adam()
    
    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        
        # Reshape input for convolution if needed
        if len(input.shape) == 2:
            side = int(np.sqrt(input.shape[1]))
            input = input.reshape(batch_size, 1, side, side)
        
        # Calculate output dimensions
        h_out = ((input.shape[2] + 2*self.padding - self.kernel_size) // self.stride) + 1
        w_out = ((input.shape[3] + 2*self.padding - self.kernel_size) // self.stride) + 1
        
        # Initialize output array
        output = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Pad input if needed
        if self.padding > 0:
            input = np.pad(input, 
                         ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 
                         mode='constant')
        
        # Store padded input for backward pass
        self.padded_input = input
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = input[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h, w] = np.sum(self.weights[c_out] * patch) + self.bias[c_out]
        
        # Store pre-activation output
        self.pre_activation = output.copy()
        
        # Apply activation function if specified
        if self.activation:
            output = self.activation(output)
        
        # Store final output
        self.output = output
        return output
    
    def backward(self, output_gradient, optimizer):
        # Apply activation gradient if needed
        if self.activation_prime:
            output_gradient = output_gradient * self.activation_prime(self.pre_activation)
            
        batch_size = output_gradient.shape[0]
        h_out, w_out = output_gradient.shape[2:]
        
        # Initialize gradients
        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.sum(output_gradient, axis=(0,2,3))
        
        # Compute weight gradients
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = self.padded_input[b, :, h_start:h_end, w_start:w_end]
                        weights_gradient[c_out] += patch * output_gradient[b, c_out, h, w]
        
        # Update parameters if optimizer is provided
        if optimizer:
            self.weights = self.weight_optimizer.update(self.weights, weights_gradient / batch_size)
            self.bias = self.bias_optimizer.update(self.bias, bias_gradient / batch_size)
        
        return input_gradient

class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, output_gradient, optimizer):
        return output_gradient.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.weight_optimizer = Adam()
        self.bias_optimizer = Adam()
        
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias
    
    def backward(self, output_gradient, optimizer):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        if optimizer:
            self.weights = self.weight_optimizer.update(self.weights, weights_gradient)
            self.bias = self.bias_optimizer.update(self.bias, np.sum(output_gradient, axis=0, keepdims=True))
        
        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, optimizer):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class BatchNorm(Layer):
    def __init__(self, input_shape, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = np.ones((1, input_shape))
        self.beta = np.zeros((1, input_shape))
        self.moving_mean = np.zeros((1, input_shape))
        self.moving_var = np.ones((1, input_shape))
        
        self.gamma_optimizer = Adam()
        self.beta_optimizer = Adam()
        
    def forward(self, input, training=True):
        self.input = input
        input_shape = input.shape[-1]
        
        if self.gamma.shape[-1] != input_shape:
            self.gamma = np.ones((1, input_shape))
            self.beta = np.zeros((1, input_shape))
            self.moving_mean = np.zeros((1, input_shape))
            self.moving_var = np.ones((1, input_shape))
        
        if training:
            mean = np.mean(input, axis=0, keepdims=True)
            var = np.var(input, axis=0, keepdims=True) + self.epsilon
            
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_var = 0.9 * self.moving_var + 0.1 * var
        else:
            mean = self.moving_mean
            var = self.moving_var
            
        self.x_norm = (input - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.x_norm + self.beta
        
    def backward(self, output_gradient, optimizer):
        gamma_grad = np.sum(output_gradient * self.x_norm, axis=0, keepdims=True)
        beta_grad = np.sum(output_gradient, axis=0, keepdims=True)
        
        if optimizer:
            self.gamma = self.gamma_optimizer.update(self.gamma, gamma_grad)
            self.beta = self.beta_optimizer.update(self.beta, beta_grad)
        
        return output_gradient * self.gamma / np.sqrt(self.moving_var + self.epsilon)
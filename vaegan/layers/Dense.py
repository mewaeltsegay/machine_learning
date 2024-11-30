import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        """
        Dense (Fully Connected) Layer
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            activation: Activation function (default: None)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights and biases using Xavier initialization
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.biases = np.zeros(output_dim)
        
        # Cache for backprop
        self.input = None
        self.output_preactivation = None
        self.output = None
        
        # Gradients
        self.d_weights = None
        self.d_biases = None

    def forward(self, input):
        """
        Forward pass
        
        Args:
            input: Input data of shape (batch_size, input_dim)
            
        Returns:
            Output data of shape (batch_size, output_dim)
        """
        # Save input for backprop
        self.input = input
        
        # Ensure input is 2D
        if len(input.shape) > 2:
            input = input.reshape(input.shape[0], -1)
            
        # Check input dimension
        if input.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, but got {input.shape[1]}")
            
        # Linear transformation
        self.output_preactivation = np.dot(input, self.weights) + self.biases
        
        # Apply activation if specified
        if self.activation is not None:
            self.output = self.activation(self.output_preactivation)
        else:
            self.output = self.output_preactivation
            
        return self.output

    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        # If we used activation, apply its derivative
        if self.activation is not None:
            grad_output = grad_output * self.activation.derivative(self.output_preactivation)
        
        # Calculate gradients
        self.d_weights = np.dot(self.input.T, grad_output)
        self.d_biases = np.sum(grad_output, axis=0)
        
        # Calculate gradient for the next layer
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input

    def get_params(self):
        """Get layer parameters"""
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_params(self, params):
        """Set layer parameters"""
        self.weights = params['weights']
        self.biases = params['biases']

    def get_gradients(self):
        """Get parameter gradients"""
        return {
            'weights': self.d_weights,
            'biases': self.d_biases
        }

    def zero_grad(self):
        """
        Zero out the gradient values
        """
        self.d_weights = None
        self.d_biases = None

import numpy as np

class ReLU:
    def __call__(self, x):
        """
        Rectified Linear Unit activation function
        f(x) = max(0, x)
        """
        return np.maximum(0, x)
    
    def derivative(self, x):
        """
        Derivative of ReLU
        f'(x) = 1 if x > 0 else 0
        """
        return np.where(x > 0, 1.0, 0.0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        """
        Leaky ReLU with customizable slope for negative values
        
        Args:
            alpha (float): Slope for negative values (default: 0.01)
        """
        self.alpha = alpha
    
    def __call__(self, x):
        """
        f(x) = x if x > 0 else alpha * x
        """
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        """
        f'(x) = 1 if x > 0 else alpha
        """
        return np.where(x > 0, 1.0, self.alpha)

class Sigmoid:
    def __call__(self, x):
        """
        Sigmoid activation function
        f(x) = 1 / (1 + e^(-x))
        """
        # Clip x to avoid overflow
        x = np.clip(x, -88.72, 88.72)  # prevents overflow in exp
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        """
        Derivative of sigmoid
        f'(x) = f(x) * (1 - f(x))
        """
        s = self.__call__(x)
        return s * (1 - s)

class Tanh:
    def __call__(self, x):
        """
        Hyperbolic tangent activation function
        f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        """
        return np.tanh(x)
    
    def derivative(self, x):
        """
        Derivative of tanh
        f'(x) = 1 - tanh^2(x)
        """
        return 1 - np.square(np.tanh(x))

class Softmax:
    def __call__(self, x):
        """
        Softmax activation function
        f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Note: Includes numerical stability improvements
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x):
        """
        Derivative of softmax
        Note: This is a simplified version that works when used with cross-entropy loss
        """
        s = self.__call__(x)
        return s * (1 - s)

class ELU:
    def __init__(self, alpha=1.0):
        """
        Exponential Linear Unit
        
        Args:
            alpha (float): Scale for negative values (default: 1.0)
        """
        self.alpha = alpha
    
    def __call__(self, x):
        """
        f(x) = x if x > 0 else alpha * (e^x - 1)
        """
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def derivative(self, x):
        """
        f'(x) = 1 if x > 0 else alpha * e^x
        """
        return np.where(x > 0, 1, self.alpha * np.exp(x))

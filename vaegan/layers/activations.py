import cupy as cp

class ReLU:
    def __call__(self, x):
        return cp.maximum(0, x)
    
    def derivative(self, x):
        return cp.where(x > 0, 1.0, 0.0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def __call__(self, x):
        return cp.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        return cp.where(x > 0, 1.0, self.alpha)

class Sigmoid:
    def __call__(self, x):
        x = cp.clip(x, -88.72, 88.72)
        return 1 / (1 + cp.exp(-x))
    
    def derivative(self, x):
        s = self.__call__(x)
        return s * (1 - s)

class Tanh:
    def __call__(self, x):
        return cp.tanh(x)
    
    def derivative(self, x):
        return 1 - cp.square(cp.tanh(x))

class Softmax:
    def __call__(self, x):
        """
        Softmax activation function
        f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Note: Includes numerical stability improvements
        """
        # Subtract max for numerical stability
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)
    
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
        return cp.where(x > 0, x, self.alpha * (cp.exp(x) - 1))
    
    def derivative(self, x):
        """
        f'(x) = 1 if x > 0 else alpha * e^x
        """
        return cp.where(x > 0, 1, self.alpha * cp.exp(x))

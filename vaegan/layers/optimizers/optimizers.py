import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer implementation
        
        Args:
            learning_rate (float): Learning rate (default: 0.001)
            beta1 (float): Exponential decay rate for first moment (default: 0.9)
            beta2 (float): Exponential decay rate for second moment (default: 0.999)
            epsilon (float): Small constant for numerical stability (default: 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        
        # Initialize dictionaries to store momentum and velocity for each parameter
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate

    def update(self, params, grads):
        """
        Update parameters using Adam optimization
        
        Args:
            params (dict): Dictionary of parameters to update
            grads (dict): Dictionary of gradients corresponding to parameters
        """
        if not self.m:  # Initialize momentum and velocity on first update
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1

        for key in params:
            if key not in grads:
                continue

            # Get gradients
            grad = grads[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grad)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params, grads):
        """Update parameters using Adam optimization"""
        if not self.m:  # Initialize momentum and velocity if first update
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        
        for key in params:
            # Update momentum and velocity
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
            
            # Update parameters
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
        
        return params
import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def initialize(self, params_shape):
        if self.m is None:
            self.m = np.zeros(params_shape)
            self.v = np.zeros(params_shape)
    
    def update(self, params, grads):
        self.initialize(params.shape)
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

class RNN:
    def __init__(self, input_size, hidden_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(input_size, hidden_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros((1, hidden_size))
        
        # Initialize optimizers
        self.optimizers = {
            'Wxh': Adam(),
            'Whh': Adam(),
            'bh': Adam()
        }
        
        self.reset_state()
    
    def reset_state(self):
        self.h = np.zeros((self.batch_size, self.hidden_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """
        Forward pass of RNN
        x shape: (time_steps, batch_size, input_size)
        Returns: (time_steps, batch_size, hidden_size)
        """
        time_steps = x.shape[0]
        outputs = np.zeros((time_steps, self.batch_size, self.hidden_size))
        self.cache = []
        
        for t in range(time_steps):
            # Get current input
            xt = x[t]
            
            # Compute hidden state
            h_raw = np.dot(xt, self.Wxh) + np.dot(self.h, self.Whh) + self.bh
            self.h = self.relu(h_raw)
            
            outputs[t] = self.h
            self.cache.append((xt, h_raw))
        
        return outputs
    
    def backward(self, dh_next, learning_rate=0.001):
        """
        Backward pass of RNN
        dh_next shape: (time_steps, batch_size, hidden_size)
        """
        time_steps = len(self.cache)
        
        dWxh = 0
        dWhh = 0
        dbh = 0
        
        dh_prev = np.zeros_like(self.h)
        
        for t in reversed(range(time_steps)):
            xt, h_raw = self.cache[t]
            dh = dh_next[t] + dh_prev
            
            # Backprop through ReLU
            dh_raw = dh * self.relu_derivative(h_raw)
            
            # Compute gradients
            dWxh += np.dot(xt.T, dh_raw)
            dWhh += np.dot(self.h.T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            # Compute gradient for next iteration
            dh_prev = np.dot(dh_raw, self.Whh.T)
        
        # Update weights using Adam optimizer
        self.Wxh = self.optimizers['Wxh'].update(self.Wxh, dWxh)
        self.Whh = self.optimizers['Whh'].update(self.Whh, dWhh)
        self.bh = self.optimizers['bh'].update(self.bh, dbh)

class DataBatcher:
    def __init__(self, X, y, batch_size, shuffle=True):
        """
        Initialize data batcher
        X shape: (total_sequences, time_steps, features)
        y shape: (total_sequences, time_steps, output_size)
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.reset()
    
    def reset(self):
        """Reset the batcher and shuffle data if needed"""
        self.current_idx = 0
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
            self.X = self.X[indices]
            self.y = self.y[indices]
    
    def has_next(self):
        """Check if there are more batches"""
        return self.current_idx < self.n_samples
    
    def next_batch(self):
        """Get next batch of data"""
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_X = self.X[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]
        
        self.current_idx = end_idx
        
        # Transpose to get (time_steps, batch_size, features)
        return (np.transpose(batch_X, (1, 0, 2)), 
                np.transpose(batch_y, (1, 0, 2)))
    
class RNNTrainer:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.rnn = RNN(input_size, hidden_size, batch_size)
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Output layer weights
        scale = np.sqrt(2.0 / (hidden_size + output_size))
        self.Wy = np.random.randn(hidden_size, output_size) * scale
        self.by = np.zeros((1, output_size))
        
        # Output layer optimizer
        self.Wy_optimizer = Adam()
        self.by_optimizer = Adam()
        
        # Initialize loss history
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, X):
        """Forward pass through RNN and output layer"""
        self.rnn_outputs = self.rnn.forward(X)
        
        time_steps = X.shape[0]
        current_batch_size = X.shape[1]
        predictions = np.zeros((time_steps, current_batch_size, self.output_size))
        
        for t in range(time_steps):
            # Linear output for regression
            predictions[t] = np.dot(self.rnn_outputs[t], self.Wy) + self.by
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """Compute MSE loss"""
        return np.mean((predictions - targets) ** 2)
    
    def compute_metrics(self, predictions, targets):
        """Compute regression metrics"""
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics
    
    def backward(self, predictions, targets):
        """Backward pass through output layer and RNN"""
        time_steps = predictions.shape[0]
        batch_size = predictions.shape[1]
        
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        drnn_outputs = np.zeros_like(self.rnn_outputs)
        
        for t in range(time_steps):
            # MSE gradient
            dout = 2 * (predictions[t] - targets[t]) / (batch_size * time_steps)
            
            dWy += np.dot(self.rnn_outputs[t].T, dout)
            dby += np.sum(dout, axis=0, keepdims=True)
            drnn_outputs[t] = np.dot(dout, self.Wy.T)
        
        # Update weights
        self.Wy = self.Wy_optimizer.update(self.Wy, dWy)
        self.by = self.by_optimizer.update(self.by, dby)
        self.rnn.backward(drnn_outputs)
    
    def train_epoch(self, data_batcher, validate=False):
        """Train/validate for one epoch"""
        epoch_metrics = {
            'mse': [],
            'rmse': [],
            'mae': []
        }
        
        data_batcher.reset()
        
        while data_batcher.has_next():
            X_batch, y_batch = data_batcher.next_batch()
            current_batch_size = X_batch.shape[1]
            
            # Adjust RNN's batch size if needed
            if current_batch_size != self.rnn.batch_size:
                self.rnn.batch_size = current_batch_size
                self.rnn.reset_state()
            
            # Forward pass
            predictions = self.forward(X_batch)
            
            # Compute metrics
            batch_metrics = self.compute_metrics(predictions, y_batch)
            for key in epoch_metrics:
                epoch_metrics[key].append(batch_metrics[key])
            
            # Backward pass (only during training)
            if not validate:
                self.backward(predictions, y_batch)
            
            # Reset state between batches
            self.rnn.reset_state()
        
        # Compute average metrics for the epoch
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def train(self, X_train, y_train, epochs, batch_size, 
              X_val=None, y_val=None, shuffle=True):
        """Train the model"""
        train_batcher = DataBatcher(X_train, y_train, batch_size, shuffle)
        val_batcher = None if X_val is None else DataBatcher(
            X_val, y_val, batch_size, False)
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_batcher)
            self.train_losses.append(train_metrics['mse'])
            
            # Validation
            val_metrics = None
            if val_batcher is not None:
                val_metrics = self.train_epoch(val_batcher, validate=True)
                self.val_losses.append(val_metrics['mse'])
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train - MSE: {train_metrics['mse']:.4f}, "
                  f"RMSE: {train_metrics['rmse']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}")
            
            if val_metrics is not None:
                print(f"Val - MSE: {val_metrics['mse']:.4f}, "
                      f"RMSE: {val_metrics['rmse']:.4f}, "
                      f"MAE: {val_metrics['mae']:.4f}")
            print("-" * 50)
    
    def predict(self, X, batch_size=None):
        """Make predictions"""
        if batch_size is None:
            batch_size = self.batch_size
        
        pred_batcher = DataBatcher(X, np.zeros_like(X), batch_size, shuffle=False)
        predictions = []
        
        while pred_batcher.has_next():
            X_batch, _ = pred_batcher.next_batch()
            current_batch_size = X_batch.shape[1]
            
            if current_batch_size != self.rnn.batch_size:
                self.rnn.batch_size = current_batch_size
                self.rnn.reset_state()
            
            batch_predictions = self.forward(X_batch)
            predictions.append(np.transpose(batch_predictions, (1, 0, 2)))
            
            self.rnn.reset_state()
        
        return np.concatenate(predictions, axis=0)

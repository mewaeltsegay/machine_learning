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

class GRU:
    def __init__(self, input_size, hidden_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Initialize weights
        # Update gate weights
        self.Wz = np.random.randn(input_size, hidden_size) * 0.01
        self.Uz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((1, hidden_size))
        
        # Reset gate weights
        self.Wr = np.random.randn(input_size, hidden_size) * 0.01
        self.Ur = np.random.randn(hidden_size, hidden_size) * 0.01
        self.br = np.zeros((1, hidden_size))
        
        # Hidden state weights
        self.Wh = np.random.randn(input_size, hidden_size) * 0.01
        self.Uh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        
        # Initialize optimizers for each weight
        self.optimizers = {
            'Wz': Adam(), 'Uz': Adam(), 'bz': Adam(),
            'Wr': Adam(), 'Ur': Adam(), 'br': Adam(),
            'Wh': Adam(), 'Uh': Adam(), 'bh': Adam()
        }
        
        self.reset_state()
    
    def reset_state(self):
        self.h = np.zeros((self.batch_size, self.hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def forward(self, x):
        """
        Forward pass of GRU
        x shape: (time_steps, batch_size, input_size)
        Returns: (time_steps, batch_size, hidden_size)
        """
        time_steps = x.shape[0]
        outputs = np.zeros((time_steps, self.batch_size, self.hidden_size))
        self.cache = []
        
        for t in range(time_steps):
            # Get current input
            xt = x[t]
            
            # Update gate
            z = self.sigmoid(np.dot(xt, self.Wz) + np.dot(self.h, self.Uz) + self.bz)
            
            # Reset gate
            r = self.sigmoid(np.dot(xt, self.Wr) + np.dot(self.h, self.Ur) + self.br)
            
            # Candidate hidden state
            h_candidate = np.tanh(np.dot(xt, self.Wh) + np.dot(r * self.h, self.Uh) + self.bh)
            
            # New hidden state
            self.h = z * self.h + (1 - z) * h_candidate
            
            outputs[t] = self.h
            self.cache.append((xt, z, r, h_candidate))
        
        return outputs
    
    def backward(self, dh_next, learning_rate=0.001):
        """
        Backward pass of GRU
        dh_next shape: (time_steps, batch_size, hidden_size)
        """
        time_steps = len(self.cache)
        
        dWz, dUz, dbz = 0, 0, 0
        dWr, dUr, dbr = 0, 0, 0
        dWh, dUh, dbh = 0, 0, 0
        
        dh_prev = np.zeros_like(self.h)
        
        for t in reversed(range(time_steps)):
            xt, z, r, h_candidate = self.cache[t]
            dh = dh_next[t] + dh_prev
            
            # Update gate derivatives
            dz = dh * (self.h - h_candidate)
            dz_input = dz * z * (1 - z)
            dWz += np.dot(xt.T, dz_input)
            dUz += np.dot(self.h.T, dz_input)
            dbz += np.sum(dz_input, axis=0, keepdims=True)
            
            # Reset gate derivatives
            dh_candidate = dh * (1 - z)
            dh_prev = dh * z + np.dot(dz_input, self.Uz.T)
            
            dr = np.dot(dh_candidate, self.Uh.T) * self.h
            dr_input = dr * r * (1 - r)
            dWr += np.dot(xt.T, dr_input)
            dUr += np.dot(self.h.T, dr_input)
            dbr += np.sum(dr_input, axis=0, keepdims=True)
            
            # Hidden state derivatives
            dh_tilde = dh_candidate * (1 - h_candidate**2)
            dWh += np.dot(xt.T, dh_tilde)
            dUh += np.dot((r * self.h).T, dh_tilde)
            dbh += np.sum(dh_tilde, axis=0, keepdims=True)
        
        # Update weights using Adam optimizer
        self.Wz = self.optimizers['Wz'].update(self.Wz, dWz)
        self.Uz = self.optimizers['Uz'].update(self.Uz, dUz)
        self.bz = self.optimizers['bz'].update(self.bz, dbz)
        
        self.Wr = self.optimizers['Wr'].update(self.Wr, dWr)
        self.Ur = self.optimizers['Ur'].update(self.Ur, dUr)
        self.br = self.optimizers['br'].update(self.br, dbr)
        
        self.Wh = self.optimizers['Wh'].update(self.Wh, dWh)
        self.Uh = self.optimizers['Uh'].update(self.Uh, dUh)
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

class GRUTrainer:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.gru = GRU(input_size, hidden_size, batch_size)
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Output layer weights
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
        
        # Output layer optimizer
        self.Wy_optimizer = Adam()
        self.by_optimizer = Adam()
        
        # Initialize loss history
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, X):
        """Forward pass through GRU and output layer"""
        self.gru_outputs = self.gru.forward(X)
        
        time_steps = X.shape[0]
        current_batch_size = X.shape[1]
        predictions = np.zeros((time_steps, current_batch_size, self.output_size))
        
        for t in range(time_steps):
            predictions[t] = np.dot(self.gru_outputs[t], self.Wy) + self.by
            # predictions[t] = self.gru.sigmoid(predictions[t])
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """
        Compute MSE loss for regression
        """
        return np.mean((predictions - targets) ** 2)
    
    def compute_metrics(self, predictions, targets):
        """
        Compute regression metrics
        """
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'loss':self.compute_loss(predictions,targets)
        }
        
        return metrics
    
    def backward(self, predictions, targets):
        """Backward pass through output layer and GRU"""
        time_steps = predictions.shape[0]
        batch_size = predictions.shape[1]
        
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dgru_outputs = np.zeros_like(self.gru_outputs)
        
        # Compute gradients
        for t in range(time_steps):
            # Binary cross-entropy gradient
            dout = (predictions[t] - targets[t]) / (batch_size * time_steps)
            
            dWy += np.dot(self.gru_outputs[t].T, dout)
            dby += np.sum(dout, axis=0, keepdims=True)
            dgru_outputs[t] = np.dot(dout, self.Wy.T)
        
        # Update weights
        self.Wy = self.Wy_optimizer.update(self.Wy, dWy)
        self.by = self.by_optimizer.update(self.by, dby)
        self.gru.backward(dgru_outputs)
    
    def train_epoch(self, data_batcher, validate=False):
        """Train/validate for one epoch"""
        epoch_metrics = {
            'loss': [],
            'mse': [],
            'mae': [],
            'rmse': []
        }
        
        data_batcher.reset()
        
        while data_batcher.has_next():
            X_batch, y_batch = data_batcher.next_batch()
            current_batch_size = X_batch.shape[1]
            
            # Adjust GRU's batch size if needed
            if current_batch_size != self.gru.batch_size:
                self.gru.batch_size = current_batch_size
                self.gru.reset_state()
            
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
            self.gru.reset_state()
        
        # Compute average metrics for the epoch
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def train(self, X_train, y_train, epochs, batch_size, 
              X_val=None, y_val=None, shuffle=True):
        """
        Train the model with batch processing and metrics tracking
        """
        # Initialize data batchers
        train_batcher = DataBatcher(X_train, y_train, batch_size, shuffle)
        val_batcher = None if X_val is None else DataBatcher(
            X_val, y_val, batch_size, False)
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_batcher)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = None
            if val_batcher is not None:
                val_metrics = self.train_epoch(val_batcher, validate=True)
                self.val_losses.append(val_metrics['loss'])
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}",
                  f"MAE: {train_metrics['mae']:.4f}",
                  f"RMSE: {train_metrics['rmse']:.4f}")
            
            if val_metrics is not None:
                print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse']:.4f}",
                      f"MAE: {train_metrics['mae']:.4f}",
                  f"RMSE: {train_metrics['rmse']:.4f}")
            print("-" * 50)
    
    def predict(self, X, batch_size=None):
        """Make predictions with batch processing"""
        if batch_size is None:
            batch_size = self.batch_size
        
        pred_batcher = DataBatcher(X, np.zeros_like(X), batch_size, shuffle=False)
        predictions = []
        
        while pred_batcher.has_next():
            X_batch, _ = pred_batcher.next_batch()
            current_batch_size = X_batch.shape[1]
            
            if current_batch_size != self.gru.batch_size:
                self.gru.batch_size = current_batch_size
                self.gru.reset_state()
            
            batch_predictions = self.forward(X_batch)
            predictions.append(np.transpose(batch_predictions, (1, 0, 2)))
            
            self.gru.reset_state()
        
        return np.concatenate(predictions, axis=0)
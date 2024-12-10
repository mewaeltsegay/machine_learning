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
    
class LSTM:
    def __init__(self, input_size, hidden_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Initialize weights
        # Forget gate weights
        self.Wf = np.random.randn(input_size, hidden_size) * 0.01
        self.Uf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate weights
        self.Wi = np.random.randn(input_size, hidden_size) * 0.01
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # Output gate weights
        self.Wo = np.random.randn(input_size, hidden_size) * 0.01
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        # Cell state weights
        self.Wc = np.random.randn(input_size, hidden_size) * 0.01
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        # Initialize optimizers for each weight
        self.optimizers = {
            'Wf': Adam(), 'Uf': Adam(), 'bf': Adam(),
            'Wi': Adam(), 'Ui': Adam(), 'bi': Adam(),
            'Wo': Adam(), 'Uo': Adam(), 'bo': Adam(),
            'Wc': Adam(), 'Uc': Adam(), 'bc': Adam()
        }
        
        self.reset_state()
    
    def reset_state(self):
        self.h = np.zeros((self.batch_size, self.hidden_size))
        self.c = np.zeros((self.batch_size, self.hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def forward(self, x):
        """
        Forward pass of LSTM
        x shape: (time_steps, batch_size, input_size)
        Returns: (time_steps, batch_size, hidden_size)
        """
        time_steps = x.shape[0]
        outputs = np.zeros((time_steps, self.batch_size, self.hidden_size))
        self.cache = []
        
        for t in range(time_steps):
            # Get current input
            xt = x[t]
            
            # Forget gate
            f = self.sigmoid(np.dot(xt, self.Wf) + np.dot(self.h, self.Uf) + self.bf)
            
            # Input gate
            i = self.sigmoid(np.dot(xt, self.Wi) + np.dot(self.h, self.Ui) + self.bi)
            
            # Output gate
            o = self.sigmoid(np.dot(xt, self.Wo) + np.dot(self.h, self.Uo) + self.bo)
            
            # Cell candidate
            c_candidate = np.tanh(np.dot(xt, self.Wc) + np.dot(self.h, self.Uc) + self.bc)
            
            # Update cell state
            self.c = f * self.c + i * c_candidate
            
            # Update hidden state
            self.h = o * np.tanh(self.c)
            
            outputs[t] = self.h
            self.cache.append((xt, f, i, o, c_candidate, self.c))
        
        return outputs
    
    def backward(self, dh_next, learning_rate=0.001):
        """
        Backward pass of LSTM
        dh_next shape: (time_steps, batch_size, hidden_size)
        """
        time_steps = len(self.cache)
        
        dWf, dUf, dbf = 0, 0, 0
        dWi, dUi, dbi = 0, 0, 0
        dWo, dUo, dbo = 0, 0, 0
        dWc, dUc, dbc = 0, 0, 0
        
        dh_prev = np.zeros_like(self.h)
        dc_prev = np.zeros_like(self.c)
        
        for t in reversed(range(time_steps)):
            xt, f, i, o, c_candidate, c = self.cache[t]
            dh = dh_next[t] + dh_prev
            dc = dc_prev + (dh * o * (1 - np.tanh(c)**2))
            
            # Output gate derivatives
            do = dh * np.tanh(c)
            do_input = do * o * (1 - o)
            dWo += np.dot(xt.T, do_input)
            dUo += np.dot(self.h.T, do_input)
            dbo += np.sum(do_input, axis=0, keepdims=True)
            
            # Input gate derivatives
            di = dc * c_candidate
            di_input = di * i * (1 - i)
            dWi += np.dot(xt.T, di_input)
            dUi += np.dot(self.h.T, di_input)
            dbi += np.sum(di_input, axis=0, keepdims=True)
            
            # Forget gate derivatives
            df = dc * self.c
            df_input = df * f * (1 - f)
            dWf += np.dot(xt.T, df_input)
            dUf += np.dot(self.h.T, df_input)
            dbf += np.sum(df_input, axis=0, keepdims=True)
            
            # Cell state derivatives
            dc_candidate = dc * i
            dc_candidate_input = dc_candidate * (1 - c_candidate**2)
            dWc += np.dot(xt.T, dc_candidate_input)
            dUc += np.dot(self.h.T, dc_candidate_input)
            dbc += np.sum(dc_candidate_input, axis=0, keepdims=True)
            
            # Update gradients for next timestep
            dh_prev = (np.dot(do_input, self.Uo.T) + 
                      np.dot(di_input, self.Ui.T) + 
                      np.dot(df_input, self.Uf.T) + 
                      np.dot(dc_candidate_input, self.Uc.T))
            dc_prev = dc * f
        
        # Update weights using Adam optimizer
        self.Wf = self.optimizers['Wf'].update(self.Wf, dWf)
        self.Uf = self.optimizers['Uf'].update(self.Uf, dUf)
        self.bf = self.optimizers['bf'].update(self.bf, dbf)
        
        self.Wi = self.optimizers['Wi'].update(self.Wi, dWi)
        self.Ui = self.optimizers['Ui'].update(self.Ui, dUi)
        self.bi = self.optimizers['bi'].update(self.bi, dbi)
        
        self.Wo = self.optimizers['Wo'].update(self.Wo, dWo)
        self.Uo = self.optimizers['Uo'].update(self.Uo, dUo)
        self.bo = self.optimizers['bo'].update(self.bo, dbo)
        
        self.Wc = self.optimizers['Wc'].update(self.Wc, dWc)
        self.Uc = self.optimizers['Uc'].update(self.Uc, dUc)
        self.bc = self.optimizers['bc'].update(self.bc, dbc)

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


class LSTMTrainer:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.lstm = LSTM(input_size, hidden_size, batch_size)
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
        """Forward pass through LSTM and output layer"""
        self.lstm_outputs = self.lstm.forward(X)
        
        time_steps = X.shape[0]
        current_batch_size = X.shape[1]
        predictions = np.zeros((time_steps, current_batch_size, self.output_size))
        
        for t in range(time_steps):
            predictions[t] = np.dot(self.lstm_outputs[t], self.Wy) + self.by
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """Compute MSE loss for regression"""
        return np.mean((predictions - targets) ** 2)
    
    def compute_metrics(self, predictions, targets):
        """Compute regression metrics"""
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'loss': self.compute_loss(predictions, targets)
        }
        
        return metrics
    
    def backward(self, predictions, targets):
        """Backward pass through output layer and LSTM"""
        time_steps = predictions.shape[0]
        batch_size = predictions.shape[1]
        
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dlstm_outputs = np.zeros_like(self.lstm_outputs)
        
        # Compute gradients
        for t in range(time_steps):
            dout = (predictions[t] - targets[t]) / (batch_size * time_steps)
            
            dWy += np.dot(self.lstm_outputs[t].T, dout)
            dby += np.sum(dout, axis=0, keepdims=True)
            dlstm_outputs[t] = np.dot(dout, self.Wy.T)
        
        # Update weights
        self.Wy = self.Wy_optimizer.update(self.Wy, dWy)
        self.by = self.by_optimizer.update(self.by, dby)
        self.lstm.backward(dlstm_outputs)
    
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
            
            # Adjust LSTM's batch size if needed
            if current_batch_size != self.lstm.batch_size:
                self.lstm.batch_size = current_batch_size
                self.lstm.reset_state()
            
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
            self.lstm.reset_state()
        
        # Compute average metrics for the epoch
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def train(self, X_train, y_train, epochs, batch_size, 
              X_val=None, y_val=None, shuffle=True):
        """Train the model with batch processing and metrics tracking"""
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
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}, "
                  f"RMSE: {train_metrics['rmse']:.4f}")
            
            if val_metrics is not None:
                print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse']:.4f}, "
                      f"MAE: {val_metrics['mae']:.4f}, "
                      f"RMSE: {val_metrics['rmse']:.4f}")
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
            
            if current_batch_size != self.lstm.batch_size:
                self.lstm.batch_size = current_batch_size
                self.lstm.reset_state()
            
            batch_predictions = self.forward(X_batch)
            predictions.append(np.transpose(batch_predictions, (1, 0, 2)))
            
            self.lstm.reset_state()
        
        return np.concatenate(predictions, axis=0)
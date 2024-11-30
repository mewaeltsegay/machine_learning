import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from optimizers.adam import Adam
import matplotlib.pyplot as plt
import os

class VAEGAN:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Initialize models with optimized hyperparameters
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        self.discriminator = Discriminator(input_shape)
        
        # Optimized learning rates and Adam parameters
        initial_lr = 0.0002  # Lower learning rate for stability
        beta1 = 0.5  # Common value for GANs
        beta2 = 0.999
        
        self.encoder_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        self.decoder_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        self.discriminator_optimizer = Adam(learning_rate=initial_lr*0.5, beta1=beta1, beta2=beta2)  # Slower discriminator
        
        # Adjusted regularization parameter
        self.l2_reg = 1e-5  # Reduced L2 regularization
        
        # Loss weights for better balance
        self.reconstruction_weight = 1.0
        self.kl_weight = 0.1  # Reduced KL weight to prevent posterior collapse
        self.adversarial_weight = 0.5
        
    def adjust_learning_rate(self, epoch, initial_lr=0.0002, decay_factor=0.5, decay_epochs=20):
        """Implements smoother learning rate decay"""
        lr = initial_lr * (decay_factor ** (epoch / decay_epochs))
        for optimizer in [self.encoder_optimizer, self.decoder_optimizer, self.discriminator_optimizer]:
            optimizer.learning_rate = lr
        return lr
    
    def l2_regularization(self):
        """Optimized L2 regularization"""
        reg_loss = 0
        models = [self.encoder, self.decoder, self.discriminator]
        
        for model in models:
            if hasattr(model, 'parameters'):
                params = model.parameters()
                # Only regularize weights, not biases
                reg_loss += sum(np.sum(p**2) for k, p in params.items() if 'W' in k)
        
        return 0.5 * self.l2_reg * reg_loss
    
    def reconstruction_loss(self, x, x_recon):
        """Binary cross-entropy loss for better gradients"""
        epsilon = 1e-8
        return -np.mean(x * np.log(x_recon + epsilon) + (1 - x) * np.log(1 - x_recon + epsilon))
    
    def kl_loss(self, mean, log_var):
        """KL divergence with gradient clipping"""
        return -0.5 * np.mean(np.clip(1 + log_var - np.square(mean) - np.exp(log_var), -20, 20))
    
    def train_step(self, batch, training=True):
        # Forward passes with noise for regularization
        z, mean, log_var = self.encoder.forward(batch, training)
        x_recon = self.decoder.forward(z, training)
        
        # Add noise to discriminator inputs for stability
        noise_level = 0.05 if training else 0
        real_input = batch + np.random.normal(0, noise_level, batch.shape) if training else batch
        fake_input = x_recon + np.random.normal(0, noise_level, x_recon.shape) if training else x_recon
        
        real_output = self.discriminator.forward(real_input, training)
        fake_output = self.discriminator.forward(fake_input, training)
        
        # Calculate weighted losses
        recon_loss = self.reconstruction_weight * self.reconstruction_loss(batch, x_recon)
        kl_loss = self.kl_weight * self.kl_loss(mean, log_var)
        reg_loss = self.l2_regularization()
        
        # Label smoothing for GAN stability
        real_labels = 0.9 if training else 1.0  # Smooth positive labels
        fake_labels = 0.0  # Keep negative labels at 0
        
        disc_loss_real = -np.mean(np.log(real_output + 1e-8)) * real_labels
        disc_loss_fake = -np.mean(np.log(1 - fake_output + 1e-8))
        disc_loss = (disc_loss_real + disc_loss_fake) * self.adversarial_weight + reg_loss
        
        gen_loss = -np.mean(np.log(fake_output + 1e-8)) * self.adversarial_weight + reg_loss
        
        if training:
            # Implement backpropagation and parameter updates here
            pass
        
        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'discriminator_loss': disc_loss,
            'generator_loss': gen_loss,
            'regularization_loss': reg_loss
        }

    def generate_samples(self, num_samples=25):
        """Generate samples from random latent vectors"""
        # Sample from normal distribution
        z = np.random.normal(0, 1, (num_samples, self.latent_dim))
        # Generate images
        samples = self.decoder.forward(z, training=False)
        return samples

    def save_samples(self, epoch, samples, save_dir='samples'):
        """Save generated samples as a grid"""
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a grid of samples
        n = int(np.sqrt(len(samples)))
        grid = np.zeros((n * 28, n * 28))
        
        for i in range(n):
            for j in range(n):
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = samples[i*n + j].reshape(28, 28)
        
        # Save the grid
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
        plt.close()

# Training script
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    import numpy as np
    
    # Load and preprocess MNIST data with normalization
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    
    # Data standardization
    x_train = (x_train - 0.5) / 0.5  # Scale to [-1, 1]
    
    # Initialize VAEGAN
    vaegan = VAEGAN()
    
    # Optimized training parameters
    batch_size = 128  # Larger batch size for better statistics
    epochs = 100
    sample_interval = 5
    
    # Training loop with warm-up
    warmup_epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Learning rate adjustment
        if epoch >= warmup_epochs:
            vaegan.adjust_learning_rate(epoch - warmup_epochs)
        
        # Shuffle data
        np.random.shuffle(x_train)
        
        epoch_losses = {
            'reconstruction_loss': 0,
            'kl_loss': 0,
            'discriminator_loss': 0,
            'generator_loss': 0,
            'regularization_loss': 0
        }
        num_batches = 0
        
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            if len(batch) < batch_size:
                continue
                
            losses = vaegan.train_step(batch)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            if i % 1000 == 0:
                print(f"Batch {i//batch_size}: ", end="")
                for key, value in losses.items():
                    print(f"{key}: {value:.4f} ", end="")
                print()
        
        # Print epoch average losses
        print(f"Epoch {epoch+1} averages:")
        for key in epoch_losses:
            avg_loss = epoch_losses[key] / num_batches
            print(f"{key}: {avg_loss:.4f}")
        
        # Generate and save samples
        if (epoch + 1) % sample_interval == 0:
            print(f"Generating samples for epoch {epoch+1}...")
            samples = vaegan.generate_samples(25)  # Generate 5x5 grid
            vaegan.save_samples(epoch + 1, samples)
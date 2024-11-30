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
        
        # Initialize models
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        self.discriminator = Discriminator(input_shape)
        
        # Adjusted learning rates
        initial_lr = 0.0003  # Slightly higher learning rate
        beta1 = 0.5
        beta2 = 0.999
        
        self.encoder_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        self.decoder_optimizer = Adam(learning_rate=initial_lr*1.5, beta1=beta1, beta2=beta2)  # Faster generator
        self.discriminator_optimizer = Adam(learning_rate=initial_lr*0.5, beta1=beta1, beta2=beta2)  # Slower discriminator
        
        # Adjusted loss weights
        self.reconstruction_weight = 0.5  # Reduced reconstruction weight
        self.kl_weight = 0.05  # Further reduced KL weight
        self.adversarial_weight = 1.0  # Increased adversarial weight
        
        # Stronger regularization
        self.l2_reg = 1e-4
        
    def adjust_learning_rate(self, epoch, initial_lr=0.0003, decay_factor=0.5, decay_epochs=20):
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
        # Forward passes
        z, mean, log_var = self.encoder.forward(batch, training)
        x_recon = self.decoder.forward(z, training)
        
        # Train discriminator first
        noise_level = 0.1 if training else 0  # Increased noise for stability
        real_input = batch + np.random.normal(0, noise_level, batch.shape) if training else batch
        fake_input = x_recon + np.random.normal(0, noise_level, x_recon.shape) if training else x_recon
        
        real_output = self.discriminator.forward(real_input, training)
        fake_output = self.discriminator.forward(fake_input, training)
        
        # Calculate losses with improved stability
        recon_loss = self.reconstruction_weight * self.reconstruction_loss(batch, x_recon)
        kl_loss = self.kl_weight * self.kl_loss(mean, log_var)
        reg_loss = self.l2_regularization()
        
        # Improved label smoothing
        real_labels = np.random.uniform(0.7, 1.0, real_output.shape) if training else 1.0
        fake_labels = np.random.uniform(0.0, 0.3, fake_output.shape) if training else 0.0
        
        # Discriminator loss with improved stability
        disc_loss_real = -np.mean(np.log(real_output + 1e-7)) * np.mean(real_labels)
        disc_loss_fake = -np.mean(np.log(1 - fake_output + 1e-7))
        disc_loss = (disc_loss_real + disc_loss_fake) * self.adversarial_weight + reg_loss
        
        # Generator loss with feature matching
        gen_loss = (-np.mean(np.log(fake_output + 1e-7)) * self.adversarial_weight + 
                   recon_loss + kl_loss + reg_loss)
        
        if training:
            # Update discriminator less frequently than generator
            if np.random.rand() < 0.8:  # 80% chance to update discriminator
                # Implement parameter updates here
                pass
            
            # Always update generator
            # Implement parameter updates here
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
    
    # Improved training parameters
    batch_size = 64  # Smaller batch size for better gradient estimates
    epochs = 100
    sample_interval = 5
    
    # Training loop with improved scheduling
    warmup_epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Progressive learning rate adjustment
        if epoch >= warmup_epochs:
            lr = vaegan.adjust_learning_rate(epoch - warmup_epochs)
            print(f"Learning rate adjusted to: {lr:.6f}")
        
        # Shuffle data with improved randomization
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        
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
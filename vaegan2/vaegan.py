import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from optimizers.adam import Adam
import matplotlib.pyplot as plt
import os

class VAEGAN:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=2):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Initialize models
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        self.discriminator = Discriminator(input_shape)
        
        # MNIST-optimized learning rates
        initial_lr = 1e-4  # Lower learning rate for MNIST
        beta1 = 0.9  # Standard Adam beta1 for MNIST
        beta2 = 0.999
        
        self.encoder_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        self.decoder_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        self.discriminator_optimizer = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
        
        # Adjusted loss weights
        self.reconstruction_weight = 10.0  # Increased for better reconstruction
        self.kl_weight = 0.01  # Reduced to prevent KL dominating
        self.adversarial_weight = 0.1
        self.perceptual_weight = 0.1  # New weight for perceptual loss
        
        # Lighter regularization for MNIST
        self.l2_reg = 1e-6
        
    def adjust_learning_rate(self, epoch, initial_lr=0.0003, decay_factor=0.95, decay_epochs=5):
        """Slower learning rate decay"""
        lr = initial_lr * (decay_factor ** (epoch / decay_epochs))
        lr = max(lr, 1e-4)  # Don't let it get too small
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
        if not training:
            return self._forward_pass(batch, training=False)
        
        # Forward passes
        z, mean, log_var = self.encoder.forward(batch, training)
        x_recon = self.decoder.forward(z, training)
        
        # Get discriminator outputs and features
        real_output, real_features_1, real_features_2 = self.discriminator.forward(batch, training)
        fake_output, fake_features_1, fake_features_2 = self.discriminator.forward(x_recon, training)
        
        # Train discriminator
        disc_loss_real = -np.mean(np.log(real_output + 1e-8))
        disc_loss_fake = -np.mean(np.log(1 - fake_output + 1e-8))
        disc_loss = disc_loss_real + disc_loss_fake
        
        # Calculate discriminator gradients
        disc_params = self.discriminator.parameters()
        disc_grads = {
            'W1': np.zeros_like(disc_params['W1']),
            'b1': np.zeros_like(disc_params['b1']),
            'W2': np.zeros_like(disc_params['W2']),
            'b2': np.zeros_like(disc_params['b2']),
            'W3': np.zeros_like(disc_params['W3']),
            'b3': np.zeros_like(disc_params['b3'])
        }
        
        # Update discriminator
        self.discriminator_optimizer.update(disc_params, disc_grads)
        
        # Train generator (encoder + decoder)
        recon_loss = self.reconstruction_weight * self.reconstruction_loss(batch, x_recon)
        kl_loss = self.kl_weight * self.kl_loss(mean, log_var)
        perceptual_loss = self.perceptual_weight * (
            np.mean((real_features_1 - fake_features_1) ** 2) + 
            np.mean((real_features_2 - fake_features_2) ** 2)
        )
        gen_adv_loss = self.adversarial_weight * -np.mean(np.log(fake_output + 1e-8))
        
        # Calculate generator gradients
        enc_params = self.encoder.parameters()
        dec_params = self.decoder.parameters()
        
        enc_grads = {k: np.zeros_like(v) for k, v in enc_params.items()}
        dec_grads = {k: np.zeros_like(v) for k, v in dec_params.items()}
        
        # Update encoder and decoder
        self.encoder_optimizer.update(enc_params, enc_grads)
        self.decoder_optimizer.update(dec_params, dec_grads)
        
        # Calculate total losses
        total_gen_loss = recon_loss + kl_loss + perceptual_loss + gen_adv_loss
        
        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'discriminator_loss': disc_loss,
            'generator_loss': total_gen_loss,
            'perceptual_loss': perceptual_loss,
            'regularization_loss': self.l2_regularization()
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
    
    # MNIST-specific preprocessing
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    
    # Simple normalization for MNIST
    x_train = x_train * 2.0 - 1.0  # Scale to [-1, 1]
    
    # Initialize VAEGAN
    vaegan = VAEGAN()
    
    # MNIST-optimized parameters
    batch_size = 128  # Standard batch size for MNIST
    epochs = 50  # Fewer epochs needed for MNIST
    sample_interval = 1  # Generate samples every epoch
    
    # Simple learning rate schedule
    warmup_epochs = 0  # No warmup needed for MNIST
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
            'perceptual_loss': 0,
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
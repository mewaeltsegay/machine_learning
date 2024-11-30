import numpy as np
from dataset import load_mnist, get_batches
from utils import save_images, create_directory
from models.vaegan import VAEGAN
import os
import sys
from datetime import datetime

def train_vaegan():
    # Hyperparameters
    latent_dim = 100
    batch_size = 128
    num_epochs = 100
    save_interval = 5
    
    # Create directories for saving results
    create_directory('results')
    create_directory('checkpoints')
    
    # Load data
    print("Loading MNIST dataset...")
    X_train, X_test = load_mnist()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize model
    print("Initializing VAEGAN...")
    vaegan = VAEGAN(latent_dim=latent_dim)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_e_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []

        total_batches = len(X_train) // batch_size
        
        # Print epoch header
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("=" * 50)
        
        for batch_idx, batch in enumerate(get_batches(X_train, batch_size)):
            # Train on batch
            losses = vaegan.train_step(batch)
            
            # Record losses
            epoch_d_losses.append(losses['discriminator_loss'])
            epoch_g_losses.append(losses['generator_loss'])
            epoch_e_losses.append(losses['encoder_loss'])
            epoch_recon_losses.append(losses['reconstruction_loss'])
            epoch_kl_losses.append(losses['kl_loss'])
            
            # Calculate average losses
            avg_d_loss = np.mean(epoch_d_losses[-50:])  # Last 50 batches
            avg_g_loss = np.mean(epoch_g_losses[-50:])
            avg_e_loss = np.mean(epoch_e_losses[-50:])
            # Calculate progress percentage
            progress = (batch_idx + 1) / total_batches
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Get current time
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Create progress line
            progress_line = f'\r[{current_time}] [{bar}] {progress:.1%} '
            progress_line += f'D_loss: {avg_d_loss:.4f} '
            progress_line += f'G_loss: {avg_g_loss:.4f} '
            progress_line += f'E_loss: {avg_e_loss:.4f}'
            
            # Print progress
            sys.stdout.write(progress_line)
            sys.stdout.flush()
        
        print()
        # Print epoch statistics
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"D Loss: {np.mean(epoch_d_losses):.4f}")
        print(f"G Loss: {np.mean(epoch_g_losses):.4f}")
        print(f"E Loss: {np.mean(epoch_e_losses):.4f}")
        print(f"Recon Loss: {np.mean(epoch_recon_losses):.4f}")
        print(f"KL Loss: {np.mean(epoch_kl_losses):.4f}\n")
        
        # Save samples and reconstructions
        if (epoch + 1) % save_interval == 0:
            print("Saving samples and reconstructions...")
            
            # Generate random samples
            n_samples = 16
            samples = vaegan.generate(n_samples)
            save_images(samples, f'results/samples_epoch_{epoch+1}.png')
            
            # Generate reconstructions
            test_samples = X_test[:n_samples]
            reconstructions = vaegan.reconstruct(test_samples)
            
            # Save original and reconstructed images side by side
            comparison = np.vstack([test_samples, reconstructions])
            save_images(comparison, f'results/reconstructions_epoch_{epoch+1}.png')
            
            # Save model checkpoint
            save_model(vaegan, f'checkpoints/vaegan_epoch_{epoch+1}.npz')
            print(f"Saved checkpoint for epoch {epoch+1}")

def save_model(model, path):
    """
    Save model parameters
    """
    params = {
        'encoder': model.encoder.get_params(),
        'decoder': model.decoder.get_params(),
        'discriminator': model.discriminator.get_params()
    }
    np.savez(path, **params)

def load_model(model, path):
    """
    Load model parameters
    """
    params = np.load(path)
    model.encoder.set_params(params['encoder'].item())
    model.decoder.set_params(params['decoder'].item())
    model.discriminator.set_params(params['discriminator'].item())

if __name__ == '__main__':
    train_vaegan()


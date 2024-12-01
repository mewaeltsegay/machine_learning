import numpy as np
import os
import matplotlib.pyplot as plt
from models.vae import VAE
from models.layers import Adam
from keras.datasets import mnist

def load_mnist():
    # Load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return x_train, x_test

def get_batches(data, batch_size):
    indices = np.random.permutation(len(data))
    data = data[indices]
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

def save_samples(vae, epoch, n_samples=10):
    # Get test samples
    _, x_test = load_mnist()
    x_test = x_test[:n_samples]
    
    # Get reconstructions
    recon, _, _ = vae.forward(x_test)
    
    # Generate random samples
    z = np.random.normal(0, 1, (n_samples, vae.latent_dim))
    generated = vae.decode(z)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original images
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Original')
    
    # Plot reconstructed images
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, i + n_samples + 1)
        plt.imshow(recon[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Reconstructed')
    
    # Plot generated samples
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, i + 2*n_samples + 1)
        plt.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Generated')
    
    plt.savefig(f'vae_output/samples_epoch_{epoch}.png')
    plt.close()

def plot_losses(recon_losses, kl_losses):
    plt.figure(figsize=(15, 5))
    
    # Plot reconstruction loss
    plt.subplot(1, 2, 1)
    plt.plot(recon_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss vs. Epoch')
    
    # Plot KL divergence
    plt.subplot(1, 2, 2)
    plt.plot(kl_losses)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('vae_output/training_losses.png')
    plt.close()

def train_vae(epochs=100, batch_size=128, latent_dim=32):
    # Create output directory
    os.makedirs('vae_output', exist_ok=True)
    
    # Initialize VAE
    vae = VAE(input_dim=784, latent_dim=latent_dim, hidden_dim=512, beta=0.0)  # Start with beta=0
    optimizer = Adam(learning_rate=0.001)
    
    # Load data
    x_train, x_test = load_mnist()
    
    # Training loop
    recon_losses = []
    kl_losses = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting VAE training...")
    
    # Beta warmup parameters
    beta_start_epoch = 10
    beta_end = 1.0
    beta_warmup_epochs = 15
    
    for epoch in range(epochs):
        epoch_recon_losses = []
        epoch_kl_losses = []
        
        # Beta warmup
        if epoch >= beta_start_epoch and epoch < beta_start_epoch + beta_warmup_epochs:
            progress = (epoch - beta_start_epoch) / beta_warmup_epochs
            vae.beta = progress * beta_end
        elif epoch >= beta_start_epoch + beta_warmup_epochs:
            vae.beta = beta_end
        
        for batch in get_batches(x_train, batch_size):
            # Forward pass
            recon_batch, mean, logvar = vae.forward(batch)
            
            # Compute losses separately
            recon_loss, kl_loss = vae.compute_loss_components(batch, recon_batch, mean, logvar)
            
            # Backward pass
            vae.backward(batch, recon_batch, mean, logvar, optimizer)
            
            epoch_recon_losses.append(recon_loss)
            epoch_kl_losses.append(kl_loss)
        
        # Compute average losses
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        total_loss = avg_recon_loss + vae.beta * avg_kl_loss
        
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        print(f'Epoch {epoch}: Recon = {avg_recon_loss:.4f}, KL = {avg_kl_loss:.4f}, '
              f'Beta = {vae.beta:.4f}, Total = {total_loss:.4f}')
        
        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        # Save samples
        if epoch < 20 and epoch % 2 == 0:
            save_samples(vae, epoch)
        elif epoch % 5 == 0:
            save_samples(vae, epoch)
    
    # Plot final loss curves
    plot_losses(recon_losses, kl_losses)
    
    return vae

if __name__ == "__main__":
    trained_vae = train_vae() 
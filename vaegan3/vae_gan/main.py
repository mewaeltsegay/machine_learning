import numpy as np
import os
import matplotlib.pyplot as plt
from models.vae import VAE
from models.auxiliary import Auxiliary
from models.gan import Discriminator
from models.layers import Adam
from utils.data_loader import load_mnist, get_batches

def plot_loss_curves(vae_losses, gan_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(vae_losses, label='VAE Loss')
    plt.plot(gan_losses, label='GAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss_curves.png')
    plt.close()

def save_samples(vae, aux, epoch, n_samples=10):
    # Set to eval mode for sampling
    vae.eval()
    
    # Get test samples and labels
    x_test = load_mnist()[:n_samples]
    y_test = np.random.randint(0, 10, n_samples)
    
    # Get reconstructions
    mu, logvar = vae.forward(x_test, y_test)
    z = vae.reparameterize(mu, logvar)
    recon = aux.forward(z, y_test)
    
    # Generate random samples
    z = np.random.normal(0, 1, (n_samples, vae.latent_dim))
    y_gen = np.random.randint(0, 10, n_samples)
    generated = aux.forward(z, y_gen)
    
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
    
    plt.savefig(f'output/samples_epoch_{epoch}.png')
    plt.close()
    
    # Set back to training mode
    vae.train()

def loss_function(recon_x, x, mu, logvar, batch_size=100):
    MSE = np.mean((recon_x - x) ** 2)
    KLD = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    KLD /= batch_size * 784
    return MSE + KLD

def train(epochs=200, batch_size=32, latent_dim=20, hidden_dim=400):
    os.makedirs('output', exist_ok=True)
    
    # Initialize models
    vae = VAE(input_dim=784, latent_dim=latent_dim, hidden_dim=hidden_dim)
    aux = Auxiliary(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=784)
    discriminator = Discriminator(input_dim=784)
    
    # Set training mode
    vae.train()
    
    # Optimizers
    vae_optimizer = Adam(learning_rate=0.0001)
    aux_optimizer = Adam(learning_rate=0.0001)
    discriminator_optimizer = Adam(learning_rate=0.0001)
    
    # Load data
    x_train = load_mnist()
    
    vae_losses = []
    gan_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_vae_losses = []
        epoch_gan_losses = []
        
        for batch_idx, batch in enumerate(get_batches(x_train, batch_size)):
            # Generate random labels for this batch
            y_batch = np.random.randint(0, 10, batch.shape[0])
            
            # VAE forward pass
            mu, logvar = vae.forward(batch, y_batch)  # mu, logvar shape: (batch_size, latent_dim)
            z = vae.reparameterize(mu, logvar)
            recon = aux.forward(z, y_batch)
            
            # Discriminator forward pass
            features_real, output_real = discriminator.forward(batch, y_batch)
            features_fake, output_fake = discriminator.forward(recon, y_batch)
            
            # Train discriminator
            real_label = np.ones((batch_size, 1))
            fake_label = np.zeros((batch_size, 1))
            
            d_loss_real = -np.mean(np.log(output_real + 1e-8))
            d_loss_fake = -np.mean(np.log(1 - output_fake + 1e-8))
            d_loss = d_loss_real + d_loss_fake
            
            # Compute discriminator gradients
            d_grad = output_fake - output_real  # Simplified gradient for binary cross entropy
            discriminator.backward(d_grad, discriminator_optimizer)
            
            # Train VAE/Auxiliary
            vae_loss = loss_function(recon, batch, mu, logvar, batch_size)
            features_fake, output_fake = discriminator.forward(recon, y_batch)
            g_loss = -np.mean(np.log(output_fake + 1e-8))
            
            # Compute generator gradients
            recon_grad = 2 * (recon - batch) / batch_size  # MSE gradient
            kld_grad_mu = mu / batch_size  # Shape: (batch_size, latent_dim)
            kld_grad_logvar = (-0.5 * (1 - np.exp(logvar))) / batch_size  # Shape: (batch_size, latent_dim)
            
            # Generator wants to maximize log(D(G(z)))
            g_grad = -1 / (output_fake + 1e-8)
            
            # Get discriminator gradients w.r.t. its input
            disc_input_grad = discriminator.compute_gradient(1)
            # Trim the label part of discriminator gradient (last 10 dimensions)
            disc_input_grad = disc_input_grad[:, :784]
            
            # Combine gradients - now shapes match (batch_size, 784)
            aux_grad = recon_grad + g_grad.reshape(-1, 1) * disc_input_grad
            aux.backward(aux_grad, aux_optimizer)
            
            # Compute VAE gradients
            kld_grad_mu = mu / batch_size  # Shape: (batch_size, latent_dim)
            kld_grad_logvar = (-0.5 * (1 - np.exp(logvar))) / batch_size  # Shape: (batch_size, latent_dim)
            
            # Ensure shapes are correct before concatenation
            kld_grad_mu = kld_grad_mu.reshape(batch_size, latent_dim)
            kld_grad_logvar = kld_grad_logvar.reshape(batch_size, latent_dim)
            
            # Combine VAE gradients
            vae_grad = np.concatenate([kld_grad_mu, kld_grad_logvar], axis=1)
            
            # Update VAE
            vae.backward(vae_grad, vae_optimizer)
            
            epoch_vae_losses.append(vae_loss)
            epoch_gan_losses.append(d_loss)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(x_train)//batch_size}]: '
                      f'VAE Loss = {vae_loss:.4f}, GAN Loss = {d_loss:.4f}')
        
        vae_losses.append(np.mean(epoch_vae_losses))
        gan_losses.append(np.mean(epoch_gan_losses))
        
        # Save samples periodically
        if epoch % 10 == 0:
            save_samples(vae, aux, epoch)
            
    # Plot final loss curves
    plot_loss_curves(vae_losses, gan_losses)

if __name__ == "__main__":
    train() 
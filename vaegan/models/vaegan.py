import cupy as cp
from layers.Dense import Dense
from layers.Conv2D import Conv2D
from layers.activations import ReLU, LeakyReLU, Sigmoid, Tanh
from layers.optimizers.optimizers import Adam
from layers.TransposedConv2D import TransposedConv2D

class Encoder:
    def __init__(self, latent_dim):
        """
        VAE Encoder network
        """
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = Conv2D(1, 64, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (64, 14, 14)
        self.conv2 = Conv2D(64, 128, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (128, 7, 7)
        self.conv3 = Conv2D(128, 256, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (256, 3, 3)
        
        # Update flatten size to match actual output: 256 * 3 * 3 = 2304
        self.conv_out_size = 256 * 3 * 3  # 2304
        
        # Mean and log variance layers
        self.fc_mu = Dense(self.conv_out_size, latent_dim)
        self.fc_logvar = Dense(self.conv_out_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = cp.exp(0.5 * logvar)
        eps = cp.random.standard_normal(mu.shape)
        return mu + eps * std

    def forward(self, x):
        # Ensure input is grayscale
        if x.shape[1] == 3:
            x = cp.mean(x, axis=1, keepdims=True)
        
        x = self.conv1.forward(x)
        
        
        x = self.conv2.forward(x)
        
        
        x = self.conv3.forward(x)
        
        
        x = x.reshape(x.shape[0], -1)
        
        
        mu = self.fc_mu.forward(x)
        logvar = self.fc_logvar.forward(x)
        z = self.reparameterize(mu, logvar)
        
        
        return z, mu, logvar
    
    def get_params(self):
        """Return the parameters of the encoder."""
        return {
            'conv1_weights': self.conv1.weights,
            'conv1_biases': self.conv1.biases,
            'conv2_weights': self.conv2.weights,
            'conv2_biases': self.conv2.biases,
            'conv3_weights': self.conv3.weights,
            'conv3_biases': self.conv3.biases,
            'fc_mu_weights': self.fc_mu.weights,
            'fc_mu_biases': self.fc_mu.biases,
            'fc_logvar_weights': self.fc_logvar.weights,
            'fc_logvar_biases': self.fc_logvar.biases
        }
    
    def set_params(self, params):
        """Set the parameters of the encoder."""
        self.conv1.weights = params['conv1_weights']
        self.conv1.biases = params['conv1_biases']
        self.conv2.weights = params['conv2_weights']
        self.conv2.biases = params['conv2_biases']
        self.conv3.weights = params['conv3_weights']
        self.conv3.biases = params['conv3_biases']
        self.fc_mu.weights = params['fc_mu_weights']
        self.fc_mu.biases = params['fc_mu_biases']
        self.fc_logvar.weights = params['fc_logvar_weights']
        self.fc_logvar.biases = params['fc_logvar_biases']

    def get_gradients(self):
        """Return the gradients of all layers."""
        return {
            'conv1_weights_grad': self.conv1.d_weights,
            'conv1_biases_grad': self.conv1.d_biases,
            'conv2_weights_grad': self.conv2.d_weights,
            'conv2_biases_grad': self.conv2.d_biases,
            'conv3_weights_grad': self.conv3.d_weights,
            'conv3_biases_grad': self.conv3.d_biases,
            'fc_mu_weights_grad': self.fc_mu.d_weights,
            'fc_mu_biases_grad': self.fc_mu.d_biases,
            'fc_logvar_weights_grad': self.fc_logvar.d_weights,
            'fc_logvar_biases_grad': self.fc_logvar.d_biases
        }

    def backward(self, z_grad, mu_grad, logvar_grad):
        """
        Compute gradients through the encoder network.
        Args:
            z_grad: gradient w.r.t. latent vector z
            mu_grad: gradient w.r.t. mean vector mu
            logvar_grad: gradient w.r.t. log variance vector logvar
        """
        # Backpropagate through reparameterization
        std = cp.exp(0.5 * logvar_grad)
        eps = cp.random.standard_normal(mu_grad.shape)
        
        # Gradients for mu and logvar
        d_mu = self.fc_mu.backward(mu_grad)
        d_logvar = self.fc_logvar.backward(logvar_grad)
        
        # Combine gradients
        grad = d_mu + d_logvar
        
        # Reshape for convolutional layers
        grad = grad.reshape(-1, 256, 3, 3)
        
        # Backpropagate through conv layers
        grad = self.conv3.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.conv1.backward(grad)
        
        return grad

class Discriminator:
    def __init__(self):
        """
        Discriminator network for MNIST
        Input: (batch_size, 1, 28, 28)
        """
        self.conv1 = Conv2D(1, 64, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (64, 14, 14)
        self.conv2 = Conv2D(64, 128, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (128, 7, 7)
        self.conv3 = Conv2D(128, 256, kernel_size=4, stride=2, padding=1, 
                           activation=LeakyReLU(0.2))  # -> (256, 3, 3)
        
        # Update flatten size to match actual output: 256 * 3 * 3 = 2304
        self.flatten_size = 256 * 3 * 3  # 2304
        
        # Dense layer with corrected input size
        self.fc = Dense(self.flatten_size, 1, activation=Sigmoid())

    def forward(self, x):
        
        
        # Ensure input is grayscale
        if x.shape[1] == 3:
            x = cp.mean(x, axis=1, keepdims=True)
        
        
        x = self.conv1.forward(x)
        
        
        x = self.conv2.forward(x)
        
        
        x = self.conv3.forward(x)
        
        
        x = x.reshape(x.shape[0], -1)
        
        
        x = self.fc.forward(x)
        
        
        return x

    def get_params(self):
        """Return the parameters of the discriminator."""
        return {
            'conv1_weights': self.conv1.weights,
            'conv1_biases': self.conv1.biases,
            'conv2_weights': self.conv2.weights,
            'conv2_biases': self.conv2.biases,
            'conv3_weights': self.conv3.weights,
            'conv3_biases': self.conv3.biases,
            'fc_weights': self.fc.weights,
            'fc_biases': self.fc.biases
        }

    def set_params(self, params):
        """Set the parameters of the discriminator."""
        self.conv1.weights = params['conv1_weights']
        self.conv1.biases = params['conv1_biases']
        self.conv2.weights = params['conv2_weights']
        self.conv2.biases = params['conv2_biases']
        self.conv3.weights = params['conv3_weights']
        self.conv3.biases = params['conv3_biases']
        self.fc.weights = params['fc_weights']
        self.fc.biases = params['fc_biases']

    def get_gradients(self):
        """Return the gradients of all layers."""
        return {
            'conv1_weights_grad': self.conv1.d_weights,
            'conv1_biases_grad': self.conv1.d_biases,
            'conv2_weights_grad': self.conv2.d_weights,
            'conv2_biases_grad': self.conv2.d_biases,
            'conv3_weights_grad': self.conv3.d_weights,
            'conv3_biases_grad': self.conv3.d_biases,
            'fc_weights_grad': self.fc.d_weights,
            'fc_biases_grad': self.fc.d_biases
        }

    def backward(self, output_grad):
        """
        Compute gradients through the discriminator network.
        Args:
            output_grad: gradient w.r.t. discriminator output
        """
        # Backpropagate through fully connected layer
        grad = self.fc.backward(output_grad)
        
        # Reshape gradient for convolutional layers
        grad = grad.reshape(-1, 256, 3, 3)
        
        # Backpropagate through conv layers
        grad = self.conv3.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.conv1.backward(grad)
        
        return grad

class Decoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        
        # Dense layer to reshape
        self.fc = Dense(latent_dim, 256 * 3 * 3,activation=ReLU())
        
        # Convolutional layers with upsampling
        self.conv1 = Conv2D(256, 128, kernel_size=3, stride=1, padding=1,activation=ReLU())
        self.conv2 = Conv2D(128, 64, kernel_size=3, stride=1, padding=1,activation=ReLU())
        self.conv3 = Conv2D(64, 32, kernel_size=3, stride=1, padding=1,activation=ReLU())
        self.conv4 = Conv2D(32, 1, kernel_size=3, stride=1, padding=1,activation=Tanh())
        

    def upsample(self, x, scale_factor=2):
        """Upsample the input using nearest neighbor"""
        batch_size, channels, height, width = x.shape
        new_height = height * scale_factor
        new_width = width * scale_factor
        
        # Reshape and repeat values
        x_reshaped = x.repeat(scale_factor, axis=2).repeat(scale_factor, axis=3)
        return x_reshaped

    def forward(self, x):
        # Input shape: (batch_size, latent_dim)
        
        # Dense layer and reshape
        x = self.fc.forward(x)
        x = x.reshape(-1, 256, 3, 3)
        
        # First conv block
        x = self.upsample(x)  # 3x3 -> 6x6
        x = self.conv1.forward(x)
        
        # Second conv block
        x = self.upsample(x)  # 6x6 -> 12x12
        x = self.conv2.forward(x)
        
        # Third conv block
        x = self.upsample(x)  # 12x12 -> 24x24
        x = self.conv3.forward(x)
        
        # Final conv block
        x = self.upsample(x)  # 24x24 -> 28x28 (we'll need to crop)
        x = self.conv4.forward(x)
        
        # Crop to desired output size (28x28)
        x = x[:, :, :28, :28]
        
        return x

    def backward(self, output_grad):
        # Pad gradient if needed
        if output_grad.shape[2:] != (48, 48):  # If gradient is for cropped output
            padded_grad = cp.pad(
                output_grad,
                ((0, 0), (0, 0), (0, 20), (0, 20)),
                mode='constant'
            )
        else:
            padded_grad = output_grad
            
        # Backward through layers
        grad = self.conv4.backward(output_grad)
        grad = self._downsample(grad)
        
        grad = self.conv3.backward(grad)
        grad = self._downsample(grad)
        
        grad = self.conv2.backward(grad)
        grad = self._downsample(grad)
        
        grad = self.conv1.backward(grad)
        grad = self._downsample(grad)
        
        # Flatten gradient for dense layer
        grad = grad.reshape(grad.shape[0], -1)
        grad = self.fc.backward(grad)
        
        return grad

    def _downsample(self, x, scale_factor=2):
        """Downsample gradient during backward pass"""
        batch_size, channels, height, width = x.shape
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        # Average pooling for downsampling
        x_reshaped = x.reshape(batch_size, channels, new_height, scale_factor, 
                             new_width, scale_factor)
        return x_reshaped.mean(axis=(3, 5))

    def get_gradients(self):
        """Get all gradients of the decoder"""
        return {
            'fc_weights_grad': self.fc.d_weights,
            'fc_biases_grad': self.fc.d_biases,
            'conv1_weights_grad': self.conv1.d_weights,
            'conv1_biases_grad': self.conv1.d_biases,
            'conv2_weights_grad': self.conv2.d_weights,
            'conv2_biases_grad': self.conv2.d_biases,
            'conv3_weights_grad': self.conv3.d_weights,
            'conv3_biases_grad': self.conv3.d_biases,
            'conv4_weights_grad': self.conv4.d_weights,
            'conv4_biases_grad': self.conv4.d_biases
        }
    def get_params(self):
        """Get all parameters of the decoder"""
        return {
            'fc': self.fc.get_params(),
            'conv1': self.conv1.get_params(),
            'conv2': self.conv2.get_params(),
            'conv3': self.conv3.get_params(),
            'conv4': self.conv4.get_params()
        }

    def set_params(self, params):
        """Set all parameters of the decoder"""
        self.fc.set_params(params['fc'])
        self.conv1.set_params(params['conv1'])
        self.conv2.set_params(params['conv2'])
        self.conv3.set_params(params['conv3'])
        self.conv4.set_params(params['conv4'])
    

class VAEGAN:
    def __init__(self, latent_dim=100):
        """
        VAEGAN model combining VAE and GAN for MNIST
        Args:
            latent_dim: dimension of the latent space
        """
        self.latent_dim = latent_dim
        
        # Initialize networks
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator()
        
        # Initialize optimizers
        self.encoder_optimizer = Adam(learning_rate=0.0002, beta1=0.5)
        self.decoder_optimizer = Adam(learning_rate=0.0002, beta1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=0.0002, beta1=0.5)

    def reconstruction_loss(self, x_recon, x):
        """
        Calculate pixel-wise reconstruction loss
        """
        if x.shape[1] == 3:  # If input is RGB, convert to grayscale
            x = cp.mean(x, axis=1, keepdims=True)
        return cp.mean(cp.square(x_recon - x))

    def kl_divergence(self, mu, logvar):
        """
        Calculate KL divergence between N(mu, var) and N(0, 1)
        """
        return -0.5 * cp.mean(1 + logvar - cp.square(mu) - cp.exp(logvar))

    def train_step(self, real_images):
        batch_size = real_images.shape[0]
        
        # Forward passes
        z, mu, logvar = self.encoder.forward(real_images)
        fake_images = self.decoder.forward(z)
        fake_preds = self.discriminator.forward(fake_images)
        real_preds = self.discriminator.forward(real_images)
        
        # Discriminator loss
        d_loss_real = -cp.mean(cp.log(real_preds + 1e-8))
        d_loss_fake = -cp.mean(cp.log(1 - fake_preds + 1e-8))
        d_loss = d_loss_real + d_loss_fake
        
        # Update discriminator
        self.update_discriminator(d_loss)
        
        # Generator (Decoder) loss
        # Generate new fake images
        z = cp.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_images = self.decoder.forward(z)
        fake_preds = self.discriminator.forward(fake_images)
        
        # Compute generator loss
        g_loss = -cp.mean(cp.log(fake_preds + 1e-8))
        
        # Backpropagate through discriminator to get gradients for fake images
        d_grad = self.discriminator.backward(fake_preds - real_preds)  # Get gradients w.r.t. fake images
        
        # Now backpropagate through decoder
        g_grad = self.decoder.backward(d_grad)
        
        # Update generator
        self.update_generator(g_loss)
        
        # ====== Train Encoder ======
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(fake_images, real_images)
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Total encoder loss
        e_loss = recon_loss + kl_loss
        
        # Compute gradients and update encoder
        e_grad = self.encoder.backward(z, mu, logvar)
        self.update_encoder(e_loss)
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'encoder_loss': e_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def update_discriminator(self, loss):
        """Update discriminator parameters"""
        params = self.discriminator.get_params()
        grads = self.discriminator.get_gradients()
        updated_params = self.discriminator_optimizer.update(params, grads)
        self.discriminator.set_params(updated_params)

    def update_generator(self, loss):
        """Update generator (decoder) parameters"""
        params = self.decoder.get_params()
        grads = self.decoder.get_gradients()
        updated_params = self.decoder_optimizer.update(params, grads)
        self.decoder.set_params(updated_params)

    def update_encoder(self, loss):
        """Update encoder parameters"""
        params = self.encoder.get_params()
        grads = self.encoder.get_gradients()
        updated_params = self.encoder_optimizer.update(params, grads)
        self.encoder.set_params(updated_params)

    def generate(self, n_samples):
        """Generate new images"""
        z = cp.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.decoder.forward(z)

    def reconstruct(self, images):
        """Reconstruct images"""
        z, _, _ = self.encoder.forward(images)
        return self.decoder.forward(z)

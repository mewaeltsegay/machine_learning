import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from optimizers.adam import Adam
from utils.data_augmentation import DataAugmentation

class VAEGAN:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Initialize models
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        self.discriminator = Discriminator(input_shape)
        
        # Initialize optimizers with learning rate scheduling
        initial_lr = 0.001
        self.encoder_optimizer = Adam(learning_rate=initial_lr)
        self.decoder_optimizer = Adam(learning_rate=initial_lr)
        self.discriminator_optimizer = Adam(learning_rate=initial_lr)
        
        # Regularization parameters
        self.l2_reg = 1e-4
        
    def adjust_learning_rate(self, epoch, initial_lr=0.001, decay_factor=0.1, decay_epochs=30):
        """Implements step decay learning rate schedule"""
        lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
        for optimizer in [self.encoder_optimizer, self.decoder_optimizer, self.discriminator_optimizer]:
            optimizer.learning_rate = lr
        return lr
    
    def l2_regularization(self):
        """Calculate L2 regularization loss for all parameters"""
        reg_loss = 0
        models = [self.encoder, self.decoder, self.discriminator]
        
        for model in models:
            if hasattr(model, 'parameters'):
                params = model.parameters()
                reg_loss += np.sum([np.sum(p**2) for p in params.values()])
        
        return 0.5 * self.l2_reg * reg_loss
    
    def reconstruction_loss(self, x, x_recon):
        return np.mean(np.square(x - x_recon))
    
    def kl_loss(self, mean, log_var):
        return -0.5 * np.mean(1 + log_var - np.square(mean) - np.exp(log_var))
    
    def train_step(self, batch, training=True):
        # Apply data augmentation
        if training:
            batch = np.array([DataAugmentation.augment(img) for img in batch])
        
        # Forward passes
        z, mean, log_var = self.encoder.forward(batch, training)
        x_recon = self.decoder.forward(z, training)
        real_output = self.discriminator.forward(batch, training)
        fake_output = self.discriminator.forward(x_recon, training)
        
        # Calculate losses with regularization
        recon_loss = self.reconstruction_loss(batch, x_recon)
        kl_loss = self.kl_loss(mean, log_var)
        reg_loss = self.l2_regularization()
        
        disc_loss_real = -np.mean(np.log(real_output + 1e-8))
        disc_loss_fake = -np.mean(np.log(1 - fake_output + 1e-8))
        disc_loss = disc_loss_real + disc_loss_fake + reg_loss
        
        gen_loss = -np.mean(np.log(fake_output + 1e-8)) + reg_loss
        
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

# Training script
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    
    # Load and preprocess MNIST data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    
    # Initialize VAEGAN
    vaegan = VAEGAN()
    
    # Training parameters
    batch_size = 64
    epochs = 100
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle and batch the data
        np.random.shuffle(x_train)
        
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            if len(batch) < batch_size:
                continue
                
            losses = vaegan.train_step(batch)
            
            if i % 1000 == 0:
                print(f"Batch {i//batch_size}: ", end="")
                for key, value in losses.items():
                    print(f"{key}: {value:.4f} ", end="")
                print() 
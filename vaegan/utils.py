import numpy as np
import matplotlib.pyplot as plt

def save_images(images, path, n_rows=4):
    """
    Save a grid of images
    """
    n_samples = len(images)
    n_cols = n_samples // n_rows
    
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        image = images[i].transpose(1, 2, 0)  # CHW to HWC
        image = (image + 1) / 2.0  # [-1, 1] to [0, 1]
        plt.imshow(image)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)
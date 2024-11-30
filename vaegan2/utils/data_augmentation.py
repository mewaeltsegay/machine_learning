import numpy as np

class DataAugmentation:
    @staticmethod
    def random_rotation(image, max_angle=15):
        angle = np.random.uniform(-max_angle, max_angle)
        # Convert angle to radians
        theta = np.deg2rad(angle)
        
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Center of rotation
        center = np.array(image.shape[:2]) / 2
        
        # Create meshgrid of coordinates
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        coordinates = np.stack([x - center[1], y - center[0]], axis=-1)
        
        # Apply rotation
        new_coords = np.dot(coordinates, R.T) + center.reshape(1, 1, 2)
        
        # Interpolate
        x_coords = new_coords[..., 0]
        y_coords = new_coords[..., 1]
        
        x0 = np.floor(x_coords).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y_coords).astype(int)
        y1 = y0 + 1
        
        # Clip coordinates
        x0 = np.clip(x0, 0, image.shape[1]-1)
        x1 = np.clip(x1, 0, image.shape[1]-1)
        y0 = np.clip(y0, 0, image.shape[0]-1)
        y1 = np.clip(y1, 0, image.shape[0]-1)
        
        # Calculate interpolation weights
        wa = (x1 - x_coords) * (y1 - y_coords)
        wb = (x1 - x_coords) * (y_coords - y0)
        wc = (x_coords - x0) * (y1 - y_coords)
        wd = (x_coords - x0) * (y_coords - y0)
        
        # Interpolate
        rotated = (wa[..., None] * image[y0, x0] +
                  wb[..., None] * image[y1, x0] +
                  wc[..., None] * image[y0, x1] +
                  wd[..., None] * image[y1, x1])
        
        return rotated
    
    @staticmethod
    def random_brightness(image, max_delta=0.2):
        delta = np.random.uniform(-max_delta, max_delta)
        return np.clip(image + delta, 0, 1)
    
    @staticmethod
    def random_contrast(image, lower=0.8, upper=1.2):
        factor = np.random.uniform(lower, upper)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 1)
    
    @staticmethod
    def augment(image):
        image = DataAugmentation.random_rotation(image)
        image = DataAugmentation.random_brightness(image)
        image = DataAugmentation.random_contrast(image)
        return image 
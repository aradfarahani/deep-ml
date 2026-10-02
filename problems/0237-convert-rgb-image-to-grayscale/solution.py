import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    try:
        img = np.array(image, dtype=np.float64)
    except:
        return -1
    
    # Validate shape - must be 3D with 3 color channels
    if img.ndim != 3 or img.shape[2] != 3:
        return -1
    
    # Check for empty image
    if img.shape[0] == 0 or img.shape[1] == 0:
        return -1
    
    # Check for valid pixel values (0-255)
    if np.any(img < 0) or np.any(img > 255):
        return -1
    
    # Apply luminosity formula: Gray = 0.299*R + 0.587*G + 0.114*B
    grayscale = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    
    # Round to integers
    grayscale = np.round(grayscale).astype(int)
    
    return grayscale.tolist()
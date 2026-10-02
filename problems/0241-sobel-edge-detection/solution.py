import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.
    
    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]
    
    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    try:
        img = np.array(image, dtype=np.float64)
    except:
        return -1
    
    # Validate dimensions (must be 2D)
    if img.ndim != 2:
        return -1
    
    # Check for minimum size (at least 3x3 for Sobel)
    if img.shape[0] < 3 or img.shape[1] < 3:
        return -1
    
    # Check for valid pixel values
    if np.any(img < 0) or np.any(img > 255):
        return -1
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # Get image dimensions
    h, w = img.shape
    
    # Initialize gradient arrays
    gx = np.zeros((h - 2, w - 2))
    gy = np.zeros((h - 2, w - 2))
    
    # Apply convolution (valid mode - no padding)
    for i in range(h - 2):
        for j in range(w - 2):
            window = img[i:i+3, j:j+3]
            gx[i, j] = np.sum(window * sobel_x)
            gy[i, j] = np.sum(window * sobel_y)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Normalize to 0-255 range
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    # Round to integers
    magnitude = np.round(magnitude).astype(int)
    
    return magnitude.tolist()
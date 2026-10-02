import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image: 2D (grayscale) or 3D (RGB) array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image
    
    Returns:
        Resized image as a nested list with values rounded to 2 decimal places
    """
    image = np.array(image, dtype=np.float64)
    old_height, old_width = image.shape[:2]
    
    # Handle grayscale vs RGB
    is_grayscale = image.ndim == 2
    if is_grayscale:
        image = image[:, :, np.newaxis]
    
    channels = image.shape[2]
    result = np.zeros((new_height, new_width, channels))
    
    # Scale factors to map output to input coordinates
    y_scale = old_height / new_height
    x_scale = old_width / new_width
    
    for i in range(new_height):
        for j in range(new_width):
            # Map to source coordinates
            y = i * y_scale
            x = j * x_scale
            
            # Get integer coordinates of the four nearest pixels
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, old_height - 1)
            x1 = min(x0 + 1, old_width - 1)
            
            # Fractional parts for weighting
            dy = y - y0
            dx = x - x0
            
            # Bilinear interpolation for each channel
            for c in range(channels):
                result[i, j, c] = (
                    image[y0, x0, c] * (1 - dx) * (1 - dy) +
                    image[y0, x1, c] * dx * (1 - dy) +
                    image[y1, x0, c] * (1 - dx) * dy +
                    image[y1, x1, c] * dx * dy
                )
    
    if is_grayscale:
        result = result[:, :, 0]
    
    return np.round(result, 2).tolist()
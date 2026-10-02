import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Validate pad_width - must be non-negative integer
    if not isinstance(pad_width, (int, np.integer)) or pad_width < 0:
        return -1
    
    # Try to convert to numpy array
    try:
        image = np.array(img)
    except:
        return -1
    
    # Validate shape - must be 2D
    if image.ndim != 2:
        return -1
    
    # Check for empty image
    if image.shape[0] == 0 or image.shape[1] == 0:
        return -1
    
    # Apply zero padding
    padded = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    return padded.astype(int).tolist()
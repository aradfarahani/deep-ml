import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    try:
        img = np.array(image)
    except:
        return -1
    
    # Validate dimensions (must be 2D or 3D)
    if img.ndim < 2 or img.ndim > 3:
        return -1
    
    # Check for empty image
    if img.shape[0] == 0 or img.shape[1] == 0:
        return -1
    
    # Check direction is valid
    if direction not in ['horizontal', 'vertical']:
        return -1
    
    if direction == 'horizontal':
        # Flip left-right (reverse along column axis)
        flipped = img[:, ::-1]
    else:
        # Flip top-bottom (reverse along row axis)
        flipped = img[::-1, :]
    
    return flipped.tolist()
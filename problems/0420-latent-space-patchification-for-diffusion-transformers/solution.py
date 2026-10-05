import numpy as np

def patchify_latent(latent: np.ndarray, patch_size: int, proj_weight: np.ndarray = None, proj_bias: np.ndarray = None) -> np.ndarray:
    """
    Convert a latent representation into a sequence of patch tokens
    for a Diffusion Transformer.

    Args:
        latent: Latent tensor of shape (C, H, W)
        patch_size: Size of each square patch (p)
        proj_weight: Optional projection matrix of shape (C*p*p, D)
        proj_bias: Optional projection bias of shape (D,)

    Returns:
        tokens: Array of shape (num_patches, patch_dim) or (num_patches, D)
                if projection is applied
    """
    latent = np.asarray(latent, dtype=float)
    C, H, W = latent.shape
    p = patch_size
    
    nH = H // p
    nW = W // p
    
    # Reshape: (C, H, W) -> (C, nH, p, nW, p)
    x = latent.reshape(C, nH, p, nW, p)
    
    # Transpose to group spatial patches together: (nH, nW, C, p, p)
    x = x.transpose(1, 3, 0, 2, 4)
    
    # Flatten patches: (nH * nW, C * p * p)
    num_patches = nH * nW
    patch_dim = C * p * p
    x = x.reshape(num_patches, patch_dim)
    
    # Apply optional linear projection
    if proj_weight is not None:
        proj_weight = np.asarray(proj_weight, dtype=float)
        x = x @ proj_weight
        if proj_bias is not None:
            proj_bias = np.asarray(proj_bias, dtype=float)
            x = x + proj_bias
    
    return x
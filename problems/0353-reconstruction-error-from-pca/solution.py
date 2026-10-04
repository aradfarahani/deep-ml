import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_samples, n_features = X.shape
    
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Compute covariance matrix (population covariance)
    cov_matrix = np.cov(X_centered, rowvar=False, ddof=0)
    
    # Handle 1D feature case
    if n_features == 1:
        cov_matrix = np.array([[cov_matrix]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    V_k = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    Z = X_centered @ V_k
    
    # Reconstruct data
    X_reconstructed = Z @ V_k.T + mean
    
    # Compute MSE
    mse = np.mean((X - X_reconstructed) ** 2)
    
    return mse
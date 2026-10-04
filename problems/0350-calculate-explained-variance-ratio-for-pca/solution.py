import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    X = np.array(X, dtype=float)
    
    # Center the data by subtracting the mean
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix with unbiased estimator
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # Compute eigenvalues (eigvalsh for symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Ensure non-negative values for numerical stability
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    exp_var_ratio = eigenvalues / total_variance
    
    return exp_var_ratio.tolist()
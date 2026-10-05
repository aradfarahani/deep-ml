import numpy as np

def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    N = A.shape[0]
    
    # Step 1: Add self-loops
    A_tilde = A + np.eye(N)
    
    # Step 2: Compute degree matrix of A_tilde
    degrees = np.sum(A_tilde, axis=1)
    
    # Step 3: Compute D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    
    # Step 4: Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    # Step 5: Aggregate and transform: A_norm @ X @ W
    H = A_norm @ X @ W
    
    # Step 6: ReLU activation
    H = np.maximum(0, H)
    
    return H
import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Update gate: controls how much of the previous state to keep
    z = sigmoid(np.dot(W_z, x) + np.dot(U_z, h_prev) + b_z)
    
    # Reset gate: controls how much of the previous state to forget
    r = sigmoid(np.dot(W_r, x) + np.dot(U_r, h_prev) + b_r)
    
    # Candidate hidden state
    h_tilde = np.tanh(np.dot(W_h, x) + np.dot(U_h, r * h_prev) + b_h)
    
    # New hidden state: interpolation between previous and candidate
    h_next = (1 - z) * h_prev + z * h_tilde
    
    return h_next
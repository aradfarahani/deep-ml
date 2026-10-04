import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """
    Apply learned positional embeddings to token embeddings.
    
    Args:
        token_embeddings: (batch_size, seq_len, d_model) array of token embeddings
        position_embedding_table: (max_seq_len, d_model) learned positional embedding lookup table
        start_pos: Starting position index (default 0)
    
    Returns:
        Array of shape (batch_size, seq_len, d_model) with positional information applied
    """
    batch_size, seq_len, d_model = token_embeddings.shape
    
    # Create position indices from start_pos to start_pos + seq_len - 1
    positions = np.arange(start_pos, start_pos + seq_len)
    
    # Look up positional embeddings from the table
    pos_embeddings = position_embedding_table[positions]  # (seq_len, d_model)
    
    # Add positional embeddings to token embeddings, broadcasting over batch dimension
    output = token_embeddings + pos_embeddings[np.newaxis, :, :]
    
    return output
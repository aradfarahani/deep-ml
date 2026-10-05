import numpy as np

def eagle_draft_forward(
    hidden_state: np.ndarray,
    token_id: int,
    embed_matrix: np.ndarray,
    fc_fuse_weight: np.ndarray,
    fc_fuse_bias: np.ndarray,
    draft_head_weight: np.ndarray,
    draft_head_bias: np.ndarray,
    lm_head_weight: np.ndarray,
    num_draft_tokens: int = 3
) -> list:
    """
    Generate draft tokens using an EAGLE-style draft model.
    """
    draft_tokens = []
    current_hidden = np.array(hidden_state, dtype=np.float64)
    current_token = token_id
    
    for _ in range(num_draft_tokens):
        # Step 1: Look up token embedding
        token_embed = embed_matrix[current_token]
        
        # Step 2: Concatenate hidden state and token embedding
        concat = np.concatenate([current_hidden, token_embed])
        
        # Step 3: Fusion layer (linear + ReLU)
        fused = np.maximum(0, fc_fuse_weight @ concat + fc_fuse_bias)
        
        # Step 4: Draft head layer (linear + ReLU)
        next_hidden = np.maximum(0, draft_head_weight @ fused + draft_head_bias)
        
        # Step 5: Project to vocabulary logits
        logits = lm_head_weight @ next_hidden
        
        # Step 6: Greedy decoding (argmax)
        next_token = int(np.argmax(logits))
        
        draft_tokens.append(next_token)
        current_hidden = next_hidden
        current_token = next_token
    
    return draft_tokens
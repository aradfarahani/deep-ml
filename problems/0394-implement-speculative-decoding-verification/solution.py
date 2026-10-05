import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    draft_probs = np.array(draft_probs, dtype=np.float64)
    target_probs = np.array(target_probs, dtype=np.float64)
    K = len(draft_tokens)
    accepted = []
    
    for i in range(K):
        token = draft_tokens[i]
        q = draft_probs[i][token]
        p = target_probs[i][token]
        
        if q == 0:
            acceptance_prob = 1.0 if p == 0 else 1.0
        else:
            acceptance_prob = min(1.0, p / q)
        
        if coin_flips[i] < acceptance_prob:
            accepted.append(token)
        else:
            # Compute adjusted distribution: max(0, p(x) - q(x)) normalized
            adjusted = np.maximum(0.0, target_probs[i] - draft_probs[i])
            total = adjusted.sum()
            if total > 0:
                adjusted_norm = adjusted / total
            else:
                adjusted_norm = target_probs[i].copy()
            
            # Sample from adjusted distribution using resample_coin
            cumsum = np.cumsum(adjusted_norm)
            resampled_token = int(np.searchsorted(cumsum, resample_coin))
            resampled_token = min(resampled_token, len(adjusted_norm) - 1)
            accepted.append(resampled_token)
            return accepted
    
    return accepted
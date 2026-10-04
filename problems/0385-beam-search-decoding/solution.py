import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    T, V = log_probs.shape
    
    # Initialize beams from the first time step
    first_scores = log_probs[0]
    top_indices = np.argsort(first_scores)[::-1][:beam_width]
    beams = [(float(first_scores[idx]), [int(idx)]) for idx in top_indices]
    
    # Expand beams for each subsequent time step
    for t in range(1, T):
        all_candidates = []
        for score, seq in beams:
            for v in range(V):
                new_score = score + float(log_probs[t][v])
                new_seq = seq + [v]
                all_candidates.append((new_score, new_seq))
        
        # Keep only the top beam_width candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]
    
    # Sort final beams and return top_k
    beams.sort(key=lambda x: x[0], reverse=True)
    result = [(seq, round(score, 4)) for score, seq in beams[:top_k]]
    return result
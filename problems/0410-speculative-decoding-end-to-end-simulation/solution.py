import numpy as np

def speculative_decode(draft_probs: list, target_probs: list, draft_tokens: list, accept_rand: list, sample_rand: list) -> list:
    """
    Simulate speculative decoding end-to-end.
    """
    K = len(draft_tokens)
    accepted = []
    
    for i in range(K):
        token = draft_tokens[i]
        p_target = np.array(target_probs[i], dtype=np.float64)
        p_draft = np.array(draft_probs[i], dtype=np.float64)
        
        p_t = p_target[token]
        p_d = p_draft[token]
        
        if p_d == 0:
            acceptance = 1.0
        else:
            acceptance = min(1.0, p_t / p_d)
        
        if accept_rand[i] < acceptance:
            accepted.append(token)
        else:
            # Rejection: sample from adjusted distribution
            adjusted = np.maximum(0.0, p_target - p_draft)
            total = np.sum(adjusted)
            if total > 0:
                adjusted = adjusted / total
            else:
                adjusted = np.ones(len(p_target)) / len(p_target)
            cumsum = np.cumsum(adjusted)
            sampled = int(np.searchsorted(cumsum, sample_rand[i]))
            sampled = min(sampled, len(adjusted) - 1)
            accepted.append(sampled)
            return accepted
    
    # All K tokens accepted: sample bonus token from target_probs[K]
    p_bonus = np.array(target_probs[K], dtype=np.float64)
    cumsum = np.cumsum(p_bonus)
    bonus = int(np.searchsorted(cumsum, sample_rand[K]))
    bonus = min(bonus, len(p_bonus) - 1)
    accepted.append(bonus)
    
    return accepted
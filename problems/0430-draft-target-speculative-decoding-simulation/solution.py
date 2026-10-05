import numpy as np

def speculative_decode(draft_probs, target_probs, draft_tokens, random_coins):
    """
    Simulate one round of speculative decoding.
    """
    K = len(draft_tokens)
    accepted_tokens = []
    num_accepted = 0
    
    for i in range(K):
        token = draft_tokens[i]
        p_draft = np.array(draft_probs[i])
        p_target = np.array(target_probs[i])
        
        p_d = p_draft[token]
        p_t = p_target[token]
        
        accept_prob = min(1.0, p_t / p_d) if p_d > 0 else 0.0
        
        if random_coins[i] < accept_prob:
            accepted_tokens.append(token)
            num_accepted += 1
        else:
            # Reject: compute adjusted distribution
            adjusted = np.maximum(p_target - p_draft, 0.0)
            total = np.sum(adjusted)
            if total > 0:
                adjusted = adjusted / total
            resampled_token = int(np.argmax(adjusted))
            accepted_tokens.append(resampled_token)
            break
    else:
        # All K tokens accepted: pick bonus token from target_probs[K]
        bonus_token = int(np.argmax(np.array(target_probs[K])))
        accepted_tokens.append(bonus_token)
    
    num_generated = len(accepted_tokens)
    acceptance_rate = round(num_accepted / K, 4)
    
    return {
        'accepted_tokens': accepted_tokens,
        'num_generated': num_generated,
        'acceptance_rate': acceptance_rate
    }
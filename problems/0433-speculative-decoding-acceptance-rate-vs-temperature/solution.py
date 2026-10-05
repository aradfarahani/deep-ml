import numpy as np

def acceptance_rate_vs_temperature(draft_logits: np.ndarray, target_logits: np.ndarray, temperatures: np.ndarray) -> list:
    """
    Compute speculative decoding expected acceptance rate at various temperatures.
    
    Args:
        draft_logits: Logits from draft model, shape (vocab_size,)
        target_logits: Logits from target model, shape (vocab_size,)
        temperatures: Array of temperature values to evaluate
    
    Returns:
        List of acceptance rates (floats rounded to 4 decimal places)
    """
    def softmax_with_temp(logits, T):
        scaled = logits / T
        scaled = scaled - np.max(scaled)
        exp_vals = np.exp(scaled)
        return exp_vals / np.sum(exp_vals)
    
    rates = []
    for T in temperatures:
        p = softmax_with_temp(draft_logits, T)
        q = softmax_with_temp(target_logits, T)
        alpha = float(np.sum(np.minimum(p, q)))
        rates.append(round(alpha, 4))
    
    return rates
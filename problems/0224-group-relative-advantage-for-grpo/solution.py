import numpy as np

def compute_group_relative_advantage(rewards: list[float]) -> list[float]:
    """
    Compute Group Relative Advantage for GRPO.
    
    Args:
        rewards: List of rewards for outputs from the same prompt
        
    Returns:
        List of normalized advantages
    """
    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    
    if std_r == 0:
        return [0.0] * len(rewards)
    
    advantages = (rewards - mean_r) / std_r
    return advantages.tolist()
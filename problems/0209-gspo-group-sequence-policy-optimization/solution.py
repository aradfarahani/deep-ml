import numpy as np

def gspo_objective(log_probs_new: list[list[float]], 
                   log_probs_old: list[list[float]], 
                   rewards: list[float], 
                   epsilon: float = 0.2) -> float:
    """
    Compute GSPO clipped objective.
    
    Args:
        log_probs_new: Log probs from new policy
        log_probs_old: Log probs from old policy
        rewards: Reward for each sequence
        epsilon: Clipping range
    
    Returns:
        Average clipped objective
    """
    G = len(rewards)
    
    # Compute advantages
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward == 0:
        advantages = np.zeros(G)
    else:
        advantages = (np.array(rewards) - mean_reward) / std_reward
    
    objectives = []
    
    for i in range(G):
        # Length-normalized sequence importance ratio
        seq_len = len(log_probs_new[i])
        log_ratio_sum = sum(log_probs_new[i][t] - log_probs_old[i][t] 
                           for t in range(seq_len))
        s_i = np.exp(log_ratio_sum / seq_len)
        
        # Unclipped objective
        obj_unclipped = s_i * advantages[i]
        
        # Clipped objective
        s_i_clipped = np.clip(s_i, 1 - epsilon, 1 + epsilon)
        obj_clipped = s_i_clipped * advantages[i]
        
        # Take minimum
        obj = np.minimum(obj_unclipped, obj_clipped)
        objectives.append(obj)
    
    return float(np.mean(objectives))
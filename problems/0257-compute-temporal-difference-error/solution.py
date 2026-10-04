import numpy as np

def compute_td_error(v_s: float, reward: float, v_s_prime: float, gamma: float, done: bool) -> float:
    """
    Compute the Temporal Difference (TD) error for a single transition.
    
    Args:
        v_s: Current state value estimate V(s)
        reward: Immediate reward received
        v_s_prime: Next state value estimate V(s')
        gamma: Discount factor (0 <= gamma <= 1)
        done: True if s' is a terminal state
    
    Returns:
        The TD error delta
    """
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * v_s_prime
    
    td_error = td_target - v_s
    return td_error
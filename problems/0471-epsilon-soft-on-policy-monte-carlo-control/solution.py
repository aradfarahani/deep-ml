import numpy as np

def epsilon_soft_mc_control(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float = 0.9,
    epsilon: float = 0.1
) -> tuple:
    """
    Epsilon-soft on-policy Monte Carlo control.
    
    Args:
        episodes: List of episodes, each is a list of (state, action, reward) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor
        epsilon: Epsilon for epsilon-soft policy
    
    Returns:
        Tuple of (Q, policy) as 2D lists rounded to 4 decimal places
    """
    Q = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    policy = np.ones((n_states, n_actions)) / n_actions
    
    for episode in episodes:
        # Compute first-visit returns by processing backward
        # Overwriting ensures we keep the return from the first visit
        G = 0.0
        first_visit_returns = {}
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            first_visit_returns[(state, action)] = G
        
        # Update Q-values for all first-visited (s, a) pairs
        for (s, a), g in first_visit_returns.items():
            returns_sum[s, a] += g
            returns_count[s, a] += 1
            Q[s, a] = returns_sum[s, a] / returns_count[s, a]
        
        # Update epsilon-soft policy for all visited states
        visited_states = set(s for (s, a) in first_visit_returns.keys())
        for s in visited_states:
            best_action = int(np.argmax(Q[s]))
            for a in range(n_actions):
                if a == best_action:
                    policy[s, a] = 1.0 - epsilon + epsilon / n_actions
                else:
                    policy[s, a] = epsilon / n_actions
    
    return np.round(Q, 4).tolist(), np.round(policy, 4).tolist()
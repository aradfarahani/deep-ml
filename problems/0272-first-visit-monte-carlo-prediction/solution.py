import numpy as np

def first_visit_mc_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using First-Visit Monte Carlo prediction.
    """
    # Store sum of returns and count for each state
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    
    for episode in episodes:
        # Extract states and rewards
        states = [s for s, r in episode]
        rewards = [r for s, r in episode]
        T = len(episode)
        
        # Compute returns backwards
        G = 0.0
        returns = []
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns.append(G)
        returns = returns[::-1]  # Reverse to get G_0, G_1, ..., G_{T-1}
        
        # Track first visits
        visited = set()
        
        for t in range(T):
            state = states[t]
            if state not in visited:
                visited.add(state)
                returns_sum[state] += returns[t]
                returns_count[state] += 1
    
    # Compute average returns
    V = np.zeros(n_states)
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]
    
    return V
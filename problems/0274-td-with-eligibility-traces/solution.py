import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(Î») with accumulating eligibility traces.
    """
    V = np.zeros(n_states)
    
    for episode in episodes:
        # Initialize eligibility traces for this episode
        e = np.zeros(n_states)
        
        T = len(episode)
        states = [s for s, r in episode]
        rewards = [r for s, r in episode]
        
        for t in range(T):
            S = states[t]
            R = rewards[t]
            
            # Next state value (0 if terminal)
            if t + 1 < T:
                S_next = states[t + 1]
                V_next = V[S_next]
            else:
                V_next = 0.0  # Terminal
            
            # TD error
            delta = R + gamma * V_next - V[S]
            
            # Update eligibility trace for current state (accumulating)
            e[S] += 1
            
            # Update all state values
            V += alpha * delta * e
            
            # Decay eligibility traces
            e *= gamma * lambd
    
    return V
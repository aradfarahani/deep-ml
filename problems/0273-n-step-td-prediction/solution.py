import numpy as np

def n_step_td_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    n: int,
    gamma: float,
    alpha: float
) -> np.ndarray:
    """
    Perform n-step TD prediction to estimate state values.
    """
    V = np.zeros(n_states)
    
    for episode in episodes:
        T = len(episode)  # Episode length
        
        # Extract states and rewards
        states = [s for s, r in episode]
        rewards = [r for s, r in episode]
        
        # Process each time step
        for t in range(T):
            # Compute n-step return G_{t:t+n}
            G = 0.0
            
            # Sum discounted rewards for up to n steps
            for k in range(n):
                if t + k < T:
                    G += (gamma ** k) * rewards[t + k]
                else:
                    break
            
            # Bootstrap from V(S_{t+n}) if we haven't reached terminal
            if t + n < T:
                G += (gamma ** n) * V[states[t + n]]
            # If t + n >= T, we've reached or passed terminal, no bootstrap needed
            
            # Update V(S_t)
            V[states[t]] += alpha * (G - V[states[t]])
    
    return V
import numpy as np

def mc_control_exploring_starts(env: dict, n_states: int, n_actions: int, gamma: float, n_episodes: int, max_steps: int = 100, seed: int = 42) -> tuple:
    """
    First-visit Monte Carlo control with exploring starts.
    
    Args:
        env: dict mapping (state, action) -> (next_state, reward, done)
        n_states: number of states
        n_actions: number of actions
        gamma: discount factor
        n_episodes: number of episodes to run
        max_steps: max steps per episode
        seed: random seed for reproducibility
    
    Returns:
        Q: np.ndarray of shape (n_states, n_actions) with action-value estimates
        policy: np.ndarray of shape (n_states,) with greedy policy
    """
    np.random.seed(seed)
    
    Q = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    policy = np.random.randint(0, n_actions, size=n_states)
    
    for _ in range(n_episodes):
        # Exploring starts: pick random state and action
        s0 = np.random.randint(0, n_states)
        a0 = np.random.randint(0, n_actions)
        
        # Generate episode
        episode = []
        s, a = s0, a0
        for _ in range(max_steps):
            if (s, a) not in env:
                break
            next_s, reward, done = env[(s, a)]
            episode.append((s, a, reward))
            if done:
                break
            s = next_s
            a = policy[s]
        
        # First-visit MC: compute returns backward
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_sum[s_t, a_t] += G
                returns_count[s_t, a_t] += 1
                Q[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]
        
        # Policy improvement: greedy w.r.t. Q
        for s in range(n_states):
            policy[s] = np.argmax(Q[s])
    
    return Q, policy
import numpy as np

def async_value_iteration(num_states: int, transitions: list, gamma: float, update_order: list) -> list:
    """
    Perform asynchronous value iteration on an MDP.
    
    Args:
        num_states: Number of states in the MDP
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        update_order: Sequence of state indices to update (in order)
    
    Returns:
        List of float values representing the value function
    """
    V = np.zeros(num_states)
    
    for s in update_order:
        actions = transitions[s]
        if len(actions) == 0:
            continue
        action_values = []
        for a in range(len(actions)):
            q = 0.0
            for (prob, next_state, reward) in actions[a]:
                q += prob * (reward + gamma * V[next_state])
            action_values.append(q)
        V[s] = max(action_values)
    
    return V.tolist()
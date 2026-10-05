import numpy as np

def policy_iteration(num_states: int, num_actions: int, transitions: list, gamma: float, theta: float = 1e-8) -> tuple:
    """
    Implement the Policy Iteration algorithm for solving MDPs.
    
    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available in each state
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation
    
    Returns:
        Tuple of (policy, values) where policy is a list of ints
        and values is a list of floats rounded to 4 decimal places
    """
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        # Policy Evaluation
        while True:
            delta = 0.0
            for s in range(num_states):
                v_old = V[s]
                a = policy[s]
                new_v = 0.0
                for prob, next_s, reward in transitions[s][a]:
                    new_v += prob * (reward + gamma * V[next_s])
                V[s] = new_v
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_s, reward in transitions[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_s])
            policy[s] = int(np.argmin(-q_values))  # argmax with smallest index tie-breaking
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy.tolist(), [round(float(v), 4) for v in V]
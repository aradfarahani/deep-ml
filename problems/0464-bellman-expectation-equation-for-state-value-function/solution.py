import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    P = np.array(P, dtype=float)
    R = np.array(R, dtype=float)
    policy = np.array(policy, dtype=float)
    
    num_states = P.shape[0]
    
    # Compute policy-weighted transition matrix P_pi[s, s']
    # P_pi[s, s'] = sum_a policy[s, a] * P[s, a, s']
    P_pi = np.einsum('sa,sat->st', policy, P)
    
    # Compute policy-weighted expected immediate reward R_pi[s]
    # R_pi[s] = sum_a policy[s, a] * sum_s' P[s, a, s'] * R[s, a, s']
    R_pi = np.einsum('sa,sat,sat->s', policy, P, R)
    
    # Solve the linear system: (I - gamma * P_pi) V = R_pi
    A = np.eye(num_states) - gamma * P_pi
    V = np.linalg.solve(A, R_pi)
    
    return V
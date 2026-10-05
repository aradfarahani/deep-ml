import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers
        actions: List of action identifiers
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a)
        rewards: Dict mapping (s, a, s') -> R(s,a,s')
        policy: Dict mapping (s, a) -> pi(a|s)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places
    """
    # Initialize Q-values to zero
    Q = {}
    for s in states:
        for a in actions:
            Q[(s, a)] = 0.0

    while True:
        delta = 0.0
        new_Q = {}
        for s in states:
            for a in actions:
                q_val = 0.0
                for s_prime in states:
                    p = transition_probs.get((s, a, s_prime), 0.0)
                    r = rewards.get((s, a, s_prime), 0.0)
                    # Compute V^pi(s') = sum over a' of pi(a'|s') * Q(s', a')
                    v_s_prime = sum(
                        policy.get((s_prime, a_prime), 0.0) * Q[(s_prime, a_prime)]
                        for a_prime in actions
                    )
                    q_val += p * (r + gamma * v_s_prime)
                new_Q[(s, a)] = q_val
                delta = max(delta, abs(new_Q[(s, a)] - Q[(s, a)]))
        Q = new_Q
        if delta < theta:
            break

    return {k: round(v, 4) for k, v in Q.items()}
import numpy as np

def generalized_policy_iteration(
    num_states: int,
    num_actions: int,
    transitions: np.ndarray,
    rewards: np.ndarray,
    discount: float,
    num_iterations: int,
    eval_sweeps: int,
    terminal_states: list
) -> dict:
    """
    Simulate Generalized Policy Iteration on a finite MDP.
    """
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    terminal_set = set(terminal_states)
    
    for _ in range(num_iterations):
        # Policy Evaluation: synchronous sweeps
        for _ in range(eval_sweeps):
            V_new = np.zeros(num_states)
            for s in range(num_states):
                if s in terminal_set:
                    V_new[s] = 0.0
                    continue
                a = policy[s]
                V_new[s] = np.sum(transitions[s, a, :] * (rewards[s, a, :] + discount * V))
            V = V_new
        
        # Policy Improvement: greedy w.r.t. current V
        for s in range(num_states):
            if s in terminal_set:
                continue
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_values[a] = np.sum(transitions[s, a, :] * (rewards[s, a, :] + discount * V))
            policy[s] = int(np.argmax(q_values))
    
    return {
        'values': [round(float(v), 4) for v in V],
        'policy': policy.tolist()
    }
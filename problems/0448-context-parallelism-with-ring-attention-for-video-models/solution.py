import numpy as np

def ring_attention_simulate(Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_devices: int) -> tuple:
    """
    Simulate Ring Attention for Context Parallelism.
    """
    seq_len, d = Q.shape
    chunk_size = seq_len // num_devices
    scale = 1.0 / np.sqrt(d)
    
    Q_chunks = [Q[i*chunk_size:(i+1)*chunk_size] for i in range(num_devices)]
    K_chunks = [K[i*chunk_size:(i+1)*chunk_size] for i in range(num_devices)]
    V_chunks = [V[i*chunk_size:(i+1)*chunk_size] for i in range(num_devices)]
    
    O_acc = [np.zeros((chunk_size, d)) for _ in range(num_devices)]
    m_acc = [np.full((chunk_size, 1), -np.inf) for _ in range(num_devices)]
    l_acc = [np.zeros((chunk_size, 1)) for _ in range(num_devices)]
    
    comm_schedule = []
    
    for step in range(num_devices):
        step_sources = []
        for device in range(num_devices):
            source = (device - step) % num_devices
            step_sources.append(source)
            
            S = Q_chunks[device] @ K_chunks[source].T * scale
            
            m_new = np.max(S, axis=1, keepdims=True)
            m_combined = np.maximum(m_acc[device], m_new)
            
            exp_old = np.exp(m_acc[device] - m_combined)
            exp_S = np.exp(S - m_combined)
            
            O_acc[device] = exp_old * O_acc[device] + exp_S @ V_chunks[source]
            l_acc[device] = exp_old * l_acc[device] + np.sum(exp_S, axis=1, keepdims=True)
            m_acc[device] = m_combined
        
        comm_schedule.append(step_sources)
    
    for device in range(num_devices):
        O_acc[device] = O_acc[device] / l_acc[device]
    
    output = np.concatenate(O_acc, axis=0)
    return np.round(output, 4), comm_schedule
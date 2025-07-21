import numpy as np

def discounted_return(rewards, gamma):
    if not (0 < gamma <= 1):
        raise ValueError('gamma must be in (0, 1]')
    rewards = np.array(rewards, dtype=np.float32)
    powers = gamma ** np.arange(len(rewards))
    return float(np.sum(powers * rewards))
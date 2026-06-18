import numpy as np

def generate(model_fn, idx, max_new_tokens: int, block_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.array(idx, dtype=np.int64)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model_fn(idx_cond)
        logits = np.asarray(logits)
        last_logits = logits[:, -1, :]
        shifted = last_logits - np.max(last_logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        B, V = probs.shape
        next_tokens = np.empty((B, 1), dtype=np.int64)
        for b in range(B):
            next_tokens[b, 0] = rng.choice(V, p=probs[b])
        idx = np.concatenate([idx, next_tokens], axis=1)
    return idx.tolist()

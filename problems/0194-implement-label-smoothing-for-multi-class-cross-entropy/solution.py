import numpy as np

def smooth_labels(y_true, num_classes, epsilon):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    N = y_true.shape[0]
    K = int(num_classes)
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    off_value = epsilon / K
    on_value = 1.0 - epsilon + off_value

    targets = np.full((N, K), off_value, dtype=np.float64)
    targets[np.arange(N), y_true] = on_value
    return targets


def _log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    max_z = np.max(z, axis=1, keepdims=True)
    logsumexp = max_z + np.log(np.sum(np.exp(z - max_z), axis=1, keepdims=True))
    return z - logsumexp


def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if logits.ndim != 2:
        raise ValueError("logits must be 2D: (N, K)")
    N, K = logits.shape
    if K != num_classes:
        raise ValueError("num_classes does not match logits shape")

    targets = smooth_labels(y_true, num_classes, epsilon)
    log_probs = _log_softmax(logits)

    loss = -np.sum(targets * log_probs, axis=1).mean()

    if round_decimals is not None:
        loss = float(np.round(loss, int(round_decimals)))

    return loss

import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.
    """
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1.0

    if normalize is None:
        return cm.astype(int).tolist()

    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = np.divide(cm, row_sums, where=row_sums > 0)
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = np.divide(cm, col_sums, where=col_sums > 0)
    elif normalize == 'all':
        total = cm.sum()
        if total > 0:
            cm = cm / total
    else:
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'")

    if round_decimals is not None:
        cm = np.round(cm, int(round_decimals))

    return cm.tolist()

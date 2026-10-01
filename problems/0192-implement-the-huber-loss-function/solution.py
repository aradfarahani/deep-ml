def huber_loss(y_true, y_pred, delta=1.0):
    if not isinstance(y_true, list):
        y_true, y_pred = [y_true], [y_pred]
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        error = abs(yt - yp)
        if error <= delta:
            total += 0.5 * error ** 2
        else:
            total += delta * (error - 0.5 * delta)
    return total / len(y_true)
	
import numpy as np

def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    Determine at each epoch whether training should stop based on validation loss.
    
    Args:
        val_losses: List of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    
    Returns:
        List of booleans indicating whether to stop at each epoch
    """
    stop_signals = []
    best_loss = None
    counter = 0
    
    for val_loss in val_losses:
        if best_loss is None:
            # First epoch
            best_loss = val_loss
            stop_signals.append(False)
        elif val_loss < best_loss - min_delta:
            # Improvement found
            best_loss = val_loss
            counter = 0
            stop_signals.append(False)
        else:
            # No improvement
            counter += 1
            if counter >= patience:
                stop_signals.append(True)
            else:
                stop_signals.append(False)
    
    return stop_signals

	
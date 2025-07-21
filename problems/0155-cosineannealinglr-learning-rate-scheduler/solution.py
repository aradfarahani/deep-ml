import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        """
        Initializes the CosineAnnealingLR scheduler.
        Args:
            initial_lr (float): The initial (maximum) learning rate.
            T_max (int): The maximum number of epochs in the cosine annealing cycle.
                         The learning rate will reach min_lr at this epoch.
            min_lr (float): The minimum learning rate.
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch,
        following a cosine annealing schedule and rounded to 4 decimal places.
        Args:
            epoch (int): The current epoch number (0-indexed).
        Returns:
            float: The calculated learning rate for the current epoch, rounded to 4 decimal places.
        """
        # Ensure epoch does no
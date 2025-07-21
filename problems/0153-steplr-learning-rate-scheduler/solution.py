class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        """
        Initializes the StepLR scheduler.
        Args:
            initial_lr (float): The initial learning rate.
            step_size (int): The period of learning rate decay.
                             Learning rate is decayed every `step_size` epochs.
            gamma (float): The multiplicative factor of learning rate decay.
                           (e.g., 0.1 for reducing LR by 10x).
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch.
        Args:
            epoch (int): The current epoch number (0-indexed).
        Returns:
            float: The calculated learning rate for the current epoch.
        """
        # Calculate the number of decays that have occurred.
        # Integer division (//) in Python aut
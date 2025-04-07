import numpy as np

def descriptive_statistics(data):
    """
    Calculate various descriptive statistics metrics for a given dataset.
    :param data: List or numpy array of numerical values
    :return: Dictionary containing mean, median, mode, variance, standard deviation,
             percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Ensure data is a numpy array for easier calculations
    data = np.array(data)

    # Mean
    mean = np.mean(data)

    # Median
    median = np.median(data)

    # Mode
    unique, counts = np.unique(data, return_counts=True)
    mode = unique[np.argmax(counts)] if len(data) > 0 else None

    # Variance
    variance = np.var(data)

    # Standard Deviation
    std_dev = np.sqrt(variance)

    # Percentiles (25th, 50th, 75th)
    percentiles = np.percentile(data, [25, 50, 75])

    # Interquartile Range (IQR)
    iqr = percentiles[2] - percentiles[0]

    # Compile results into a dictionary
    stats_dict = {
        "mean"
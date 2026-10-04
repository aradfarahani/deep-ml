import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    data = np.array(data, dtype=float)
    result = data.copy()
    
    for col in range(data.shape[1]):
        col_data = data[:, col]
        mask = np.isnan(col_data)
        
        if not np.any(mask):
            continue
            
        valid_data = col_data[~mask]
        
        if len(valid_data) == 0:
            continue
        
        if strategy == 'mean':
            fill_value = np.mean(valid_data)
        elif strategy == 'median':
            fill_value = np.median(valid_data)
        elif strategy == 'mode':
            unique, counts = np.unique(valid_data, return_counts=True)
            max_count = np.max(counts)
            modes = unique[counts == max_count]
            fill_value = np.min(modes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result[mask, col] = fill_value
    
    return result
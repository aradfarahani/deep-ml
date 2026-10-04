import numpy as np

def volume_bars_sampling(prices: np.ndarray, volumes: np.ndarray, volume_threshold: float) -> list:
    """
    Generate volume bars from tick/trade data.
    
    Args:
        prices: Array of trade prices for each trade/tick
        volumes: Array of trade volumes corresponding to each price
        volume_threshold: Volume threshold that triggers formation of a new bar
    
    Returns:
        List of bars, where each bar is [open, high, low, close, total_volume]
        All values rounded to 4 decimal places
    """
    if len(prices) == 0 or len(volumes) == 0:
        return []
    
    bars = []
    cum_volume = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_close = None
    
    for i in range(len(prices)):
        price = float(prices[i])
        volume = float(volumes[i])
        
        # Start a new bar if needed
        if bar_open is None:
            bar_open = price
            bar_high = price
            bar_low = price
        
        # Update bar statistics
        bar_high = max(bar_high, price)
        bar_low = min(bar_low, price)
        bar_close = price
        cum_volume += volume
        
        # Check if threshold is reached
        if cum_volume >= volume_threshold:
            bars.append([round(bar_open, 4), round(bar_high, 4), round(bar_low, 4), 
                        round(bar_close, 4), round(cum_volume, 4)])
            # Reset for next bar
            cum_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
    
    # Handle remaining data (incomplete bar)
    if cum_volume > 0 and bar_open is not None:
        bars.append([round(bar_open, 4), round(bar_high, 4), round(bar_low, 4), 
                    round(bar_close, 4), round(cum_volume, 4)])
    
    return bars
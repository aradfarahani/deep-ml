def dollar_bars(trades: list, dollar_threshold: float) -> list:
    """
    Generate dollar bars from trade data.
    
    Args:
        trades: List of tuples (price, volume) representing individual trades
        dollar_threshold: Dollar amount threshold for creating a new bar
        
    Returns:
        List of dollar bars as tuples (open, high, low, close, volume, dollar_value)
    """
    if not trades or dollar_threshold <= 0:
        return []
    
    bars = []
    current_bar_trades = []
    accumulated_dollars = 0.0
    
    for price, volume in trades:
        dollar_value = price * volume
        accumulated_dollars += dollar_value
        current_bar_trades.append((price, volume))
        
        if accumulated_dollars >= dollar_threshold:
            prices = [t[0] for t in current_bar_trades]
            volumes = [t[1] for t in current_bar_trades]
            
            bar = (
                round(prices[0], 4),
                round(max(prices), 4),
                round(min(prices), 4),
                round(prices[-1], 4),
                round(sum(volumes), 4),
                round(accumulated_dollars, 4)
            )
            bars.append(bar)
            
            current_bar_trades = []
            accumulated_dollars = 0.0
    
    return bars
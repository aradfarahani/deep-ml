def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    if not values:
        return []
    
    # Create mapping from category to integer based on position in order
    order_map = {category: idx for idx, category in enumerate(order)}
    
    # Encode each value using the mapping
    encoded = []
    for val in values:
        if val in order_map:
            encoded.append(order_map[val])
        else:
            encoded.append(-1)
    
    return encoded
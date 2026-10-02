def birthday_problem(n: int, days: int = 365) -> float:
    """
    Calculate the probability that at least two people share the same birthday.
    
    Args:
        n: Number of people in the group
        days: Number of days in a year (default 365)
    
    Returns:
        float: Probability of at least one shared birthday, rounded to 4 decimal places
    """
    # Edge cases
    if n > days:
        # Pigeonhole principle: guaranteed match
        return 1.0
    if n <= 1:
        # Cannot have a match with 0 or 1 person
        return 0.0
    
    # Calculate probability of NO matches
    # P(no match) = (days/days) * ((days-1)/days) * ((days-2)/days) * ... * ((days-n+1)/days)
    p_no_match = 1.0
    for i in range(n):
        p_no_match *= (days - i) / days
    
    # P(at least one match) = 1 - P(no match)
    return round(1 - p_no_match, 4)
def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    if not comparisons:
        return {
            'results': [],
            'win_rate_a': 0.0,
            'win_rate_b': 0.0,
            'tie_rate': 0.0,
            'avg_margin': 0.0
        }
    
    # Normalize weights
    total_weight = sum(criteria_weights.values())
    
    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    margins = []
    
    for comp in comparisons:
        # Compute weighted scores
        weighted_a = sum(comp['scores_a'][c] * criteria_weights[c] for c in criteria_weights) / total_weight
        weighted_b = sum(comp['scores_b'][c] * criteria_weights[c] for c in criteria_weights) / total_weight
        
        margin = weighted_a - weighted_b
        abs_margin = abs(margin)
        
        if abs_margin <= tie_threshold:
            winner = 'tie'
            ties += 1
        elif margin > 0:
            winner = 'A'
            wins_a += 1
        else:
            winner = 'B'
            wins_b += 1
        
        results.append({
            'id': comp['id'],
            'winner': winner,
            'margin': round(abs_margin, 4)
        })
        margins.append(abs_margin)
    
    n = len(comparisons)
    return {
        'results': results,
        'win_rate_a': round(wins_a / n, 4),
        'win_rate_b': round(wins_b / n, 4),
        'tie_rate': round(ties / n, 4),
        'avg_margin': round(sum(margins) / n, 4)
    }
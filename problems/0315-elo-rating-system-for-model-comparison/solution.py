import numpy as np

def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    """
    Update Elo ratings based on pairwise comparison results.
    
    Args:
        ratings: Dictionary mapping model names to their current Elo ratings
        matches: List of tuples (model_a, model_b, result) where result is 'a', 'b', or 'draw'
        k_factor: The K-factor controlling rating update magnitude
    
    Returns:
        Dictionary with updated ratings for all models
    """
    # Copy ratings to avoid modifying original
    updated_ratings = {k: float(v) for k, v in ratings.items()}
    
    for model_a, model_b, result in matches:
        r_a = updated_ratings[model_a]
        r_b = updated_ratings[model_b]
        
        # Calculate expected scores using logistic function
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        e_b = 1.0 - e_a
        
        # Determine actual scores based on result
        if result == 'a':
            s_a, s_b = 1.0, 0.0
        elif result == 'b':
            s_a, s_b = 0.0, 1.0
        else:  # draw
            s_a, s_b = 0.5, 0.5
        
        # Update ratings
        updated_ratings[model_a] = r_a + k_factor * (s_a - e_a)
        updated_ratings[model_b] = r_b + k_factor * (s_b - e_b)
    
    return updated_ratings
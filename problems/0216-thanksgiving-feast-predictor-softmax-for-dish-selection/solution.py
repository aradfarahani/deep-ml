import math

def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.
    
    Args:
        preference_scores: List of preference scores for each dish
        
    Returns:
        List of probabilities for each dish
    """
    # Subtract max for numerical stability
    max_score = max(preference_scores)
    exp_scores = [math.exp(score - max_score) for score in preference_scores]
    
    # Sum of all exponentials
    sum_exp = sum(exp_scores)
    
    # Calculate softmax probabilities
    probabilities = [exp_score / sum_exp for exp_score in exp_scores]
    
    return probabilities
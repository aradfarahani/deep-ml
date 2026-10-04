def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    if not predictions or not predictions[0]:
        return []
    
    n_classifiers = len(predictions)
    n_samples = len(predictions[0])
    
    final_predictions = []
    
    for j in range(n_samples):
        # Count votes for each class for this sample
        vote_counts = {}
        for i in range(n_classifiers):
            pred = predictions[i][j]
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        # Find class with maximum votes
        max_votes = max(vote_counts.values())
        # Get all classes with max votes
        winners = [cls for cls, count in vote_counts.items() if count == max_votes]
        # Return smallest class label in case of tie
        final_predictions.append(min(winners))
    
    return final_predictions
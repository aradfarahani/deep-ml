import numpy as np

def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.
    
    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion
    
    Returns:
        Dictionary with evaluation results
    """
    judge_scores = np.array(judge_scores)
    criteria_weights = np.array(criteria_weights)
    
    # Average score per criterion across all judges
    criterion_avg = np.mean(judge_scores, axis=0)
    
    # Weighted overall score
    weighted_score = np.sum(criterion_avg * criteria_weights)
    
    # Normalized score (0-1 range)
    normalized_score = weighted_score / max_score
    
    # Pass/fail status
    pass_status = normalized_score >= passing_threshold
    
    # Inter-judge agreement: 1 - (average std / max_possible_std)
    # Max possible std for scores in [0, max_score] is max_score/2
    criterion_std = np.std(judge_scores, axis=0, ddof=0)
    avg_std = np.mean(criterion_std)
    max_possible_std = max_score / 2
    judge_agreement = max(0.0, 1.0 - (avg_std / max_possible_std))
    
    return {
        'weighted_score': round(float(weighted_score), 4),
        'normalized_score': round(float(normalized_score), 4),
        'criterion_scores': [round(float(x), 4) for x in criterion_avg],
        'pass_status': bool(pass_status),
        'judge_agreement': round(float(judge_agreement), 4)
    }
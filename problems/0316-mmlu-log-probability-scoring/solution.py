import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    log_probs = np.array(log_probs)
    n_questions = len(log_probs)
    
    # Predictions: argmax of log-probabilities
    predictions = np.argmax(log_probs, axis=1).tolist()
    
    # Calculate accuracy
    correct = sum(1 for pred, ans in zip(predictions, correct_answers) if pred == ans)
    accuracy = correct / n_questions
    
    # Convert log-probabilities to probabilities using softmax with numerical stability
    max_log_probs = np.max(log_probs, axis=1, keepdims=True)
    exp_log_probs = np.exp(log_probs - max_log_probs)
    probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
    
    # Calculate average probability assigned to correct answer
    correct_probs = [probs[i, correct_answers[i]] for i in range(n_questions)]
    avg_correct_prob = np.mean(correct_probs)
    
    return {
        'accuracy': round(accuracy, 4),
        'predictions': predictions,
        'avg_correct_prob': round(float(avg_correct_prob), 4)
    }
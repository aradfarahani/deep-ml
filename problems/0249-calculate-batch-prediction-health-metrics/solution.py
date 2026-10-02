def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not predictions:
        return {}
    
    total = len(predictions)
    successful = [p for p in predictions if p.get('status') == 'success']
    success_count = len(successful)
    
    success_rate = (success_count / total) * 100
    
    if success_count == 0:
        return {
            'success_rate': round(success_rate, 2),
            'avg_confidence': 0.0,
            'low_confidence_rate': 0.0
        }
    
    confidences = [p.get('confidence', 0.0) for p in successful]
    avg_confidence = sum(confidences) / len(confidences) * 100
    
    low_conf_count = sum(1 for c in confidences if c < confidence_threshold)
    low_confidence_rate = (low_conf_count / success_count) * 100
    
    return {
        'success_rate': round(success_rate, 2),
        'avg_confidence': round(avg_confidence, 2),
        'low_confidence_rate': round(low_confidence_rate, 2)
    }
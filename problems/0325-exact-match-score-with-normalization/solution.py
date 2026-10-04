import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    def normalize(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace: collapse multiple spaces and strip
        text = ' '.join(text.split())
        return text
    
    if len(predictions) == 0:
        return 0.0
    
    matches = sum(1 for pred, ref in zip(predictions, references) 
                  if normalize(pred) == normalize(ref))
    
    return matches / len(predictions)
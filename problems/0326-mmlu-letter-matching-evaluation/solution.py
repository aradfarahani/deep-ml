import re
from collections import defaultdict

def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    """
    valid_letters = {'A', 'B', 'C', 'D'}
    
    def extract_letter(text: str) -> str:
        """Extract answer letter from model output."""
        text = text.strip()
        if not text:
            return None
        
        # Pattern 1: Single letter
        if len(text) == 1 and text.upper() in valid_letters:
            return text.upper()
        
        # Pattern 2: Letter with punctuation like '(A)', 'A.', 'A)'
        match = re.match(r'^\(?([A-Da-d])[\.\)]?\)?$', text)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: 'The answer is X' or 'Answer: X' patterns
        match = re.search(r'(?:answer|choice)[:\s]*(?:is\s*)?[\(]?([A-Da-d])[\.\)]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 4: First character if it's a valid letter
        if text[0].upper() in valid_letters:
            return text[0].upper()
        
        return None
    
    total_questions = len(model_outputs)
    total_correct = 0
    valid_responses = 0
    
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    
    for output, truth, subject in zip(model_outputs, ground_truth, subjects):
        predicted = extract_letter(output)
        subject_total[subject] += 1
        
        if predicted is not None:
            valid_responses += 1
            if predicted == truth.upper():
                total_correct += 1
                subject_correct[subject] += 1
    
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    valid_response_rate = valid_responses / total_questions if total_questions > 0 else 0.0
    
    subject_accuracy = {}
    for subject in sorted(set(subjects)):
        if subject_total[subject] > 0:
            subject_accuracy[subject] = round(subject_correct[subject] / subject_total[subject], 4)
        else:
            subject_accuracy[subject] = 0.0
    
    return {
        'overall_accuracy': round(overall_accuracy, 4),
        'subject_accuracy': subject_accuracy,
        'valid_response_rate': round(valid_response_rate, 4),
        'total_correct': total_correct,
        'total_questions': total_questions
    }
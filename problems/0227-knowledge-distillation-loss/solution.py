import numpy as np

def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 1.0
) -> float:
    """
    Compute knowledge distillation loss.
    
    L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Softmax temperature (higher = softer)
        
    Returns:
        Distillation loss
    """
    def softmax_temp(logits, T):
        exp_logits = np.exp((logits - np.max(logits)) / T)
        return exp_logits / np.sum(exp_logits)
    
    student_probs = softmax_temp(student_logits, temperature)
    teacher_probs = softmax_temp(teacher_logits, temperature)
    
    eps = 1e-10
    kl = np.sum(teacher_probs * np.log((teacher_probs + eps) / (student_probs + eps)))
    
    return (temperature ** 2) * kl
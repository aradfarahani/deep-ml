import numpy as np

def self_critique_loss(logp_original: np.ndarray, logp_revised: np.ndarray, harm_scores: np.ndarray, lambda_penalty: float = 1.0, margin: float = 0.0) -> float:
    logp_original = np.asarray(logp_original, dtype=float)
    logp_revised = np.asarray(logp_revised, dtype=float)
    harm_scores = np.asarray(harm_scores, dtype=float)

    nll = -logp_revised
    gap = logp_original - logp_revised + margin
    hinge = np.maximum(0.0, gap)
    penalty = lambda_penalty * harm_scores * hinge

    per_example = nll + penalty
    return float(np.mean(per_example))
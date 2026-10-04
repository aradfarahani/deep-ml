import numpy as np
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    if len(candidate) == 0:
        return 0.0
    
    c = len(candidate)
    
    # Find closest reference length (if tie, pick shorter)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))
    
    # Brevity penalty
    bp = 1.0 if c >= r else np.exp(1 - r / c)
    
    precisions = []
    for n in range(1, max_n + 1):
        if c < n:
            precisions.append(0.0)
            continue
        
        # Candidate n-grams
        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(c - n + 1))
        
        # Max reference counts
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
        
        # Clipped precision
        clipped = sum(min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        
        precisions.append(clipped / total if total > 0 else 0.0)
    
    if 0.0 in precisions:
        return 0.0
    
    log_avg = sum(np.log(p) for p in precisions) / max_n
    
    return float(bp * np.exp(log_avg))
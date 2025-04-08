

import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    if not reference or not candidate:
        raise ValueError("Reference and candidate cannot be empty")
    
    # Tokenize and count
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Counter for unigram for reference and candidate 
    ref_counts = Counter(ref_tokens) 
    cand_counts = Counter(cand_tokens)
    
    # Calculate matches
    num_matches = sum((ref_counts & cand_counts).values()) # Number of matching words in candidate and reference 
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)  

    # Unigram Precision and Recall 
    precision = num_matches / cand_len if cand_len > 0 else 0 # Avoiding Division by zero
    recall = num_matches / ref_len if ref_len > 0 else 0 # Avoiding Division by zero 
    
    if num_matches == 0:
        return 0.0
    
    fmean = (precision * recall) / (alpha * p
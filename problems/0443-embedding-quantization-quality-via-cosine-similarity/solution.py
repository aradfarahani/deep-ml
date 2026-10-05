import numpy as np

def embedding_quantization_quality(embeddings: np.ndarray, num_bits: int = 8) -> dict:
    """
    Measure the quality of uniform embedding quantization using cosine similarity.
    
    Args:
        embeddings: Input embeddings of shape (n, d)
        num_bits: Number of bits for quantization
    
    Returns:
        Dictionary with mean_cosine_similarity, min_cosine_similarity,
        and max_cosine_similarity (each rounded to 6 decimal places)
    """
    n, d = embeddings.shape
    num_levels = (2 ** num_bits) - 1
    cosine_similarities = np.zeros(n)
    
    for i in range(n):
        vec = embeddings[i]
        min_val = np.min(vec)
        max_val = np.max(vec)
        
        if max_val - min_val < 1e-12:
            # Constant vector
            if np.linalg.norm(vec) > 1e-12:
                cosine_similarities[i] = 1.0
            else:
                cosine_similarities[i] = 0.0
            continue
        
        scale = (max_val - min_val) / num_levels
        quantized = np.round((vec - min_val) / scale)
        quantized = np.clip(quantized, 0, num_levels)
        dequantized = quantized * scale + min_val
        
        dot_product = np.dot(vec, dequantized)
        norm_orig = np.linalg.norm(vec)
        norm_deq = np.linalg.norm(dequantized)
        
        if norm_orig < 1e-12 or norm_deq < 1e-12:
            cosine_similarities[i] = 0.0
        else:
            cosine_similarities[i] = dot_product / (norm_orig * norm_deq)
    
    return {
        "mean_cosine_similarity": round(float(np.mean(cosine_similarities)), 6),
        "min_cosine_similarity": round(float(np.min(cosine_similarities)), 6),
        "max_cosine_similarity": round(float(np.max(cosine_similarities)), 6)
    }
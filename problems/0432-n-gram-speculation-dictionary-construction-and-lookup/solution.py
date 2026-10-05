def ngram_speculation(token_ids: list[int], n: int, context: tuple[int, ...], k: int) -> list[tuple[int, float]]:
    """
    Build an n-gram speculation dictionary from token_ids and look up
    the top-k most probable next tokens for the given context.
    """
    from collections import defaultdict
    
    # Build n-gram counts: prefix -> {next_token: count}
    ngram_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(token_ids) - n + 1):
        prefix = tuple(token_ids[i:i + n - 1])
        next_token = token_ids[i + n - 1]
        ngram_counts[prefix][next_token] += 1
    
    # Look up the context
    if context not in ngram_counts:
        return []
    
    counts = ngram_counts[context]
    total = sum(counts.values())
    
    # Compute probabilities
    probs = [(token_id, count / total) for token_id, count in counts.items()]
    
    # Sort by probability descending, then by token_id ascending for ties
    probs.sort(key=lambda x: (-x[1], x[0]))
    
    # Return top-k
    return probs[:k]
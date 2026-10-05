def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.

    Args:
        prompts: List of tokenized prompts (list of list of ints).
        cache: List of cached prefixes (list of list of ints).

    Returns:
        Dictionary with 'hit_rate', 'cached_tokens', and 'total_tokens'.
    """
    total_tokens = 0
    cached_tokens = 0

    for prompt in prompts:
        total_tokens += len(prompt)
        best_match = 0
        for cached_prefix in cache:
            prefix_len = len(cached_prefix)
            if prefix_len <= len(prompt) and prompt[:prefix_len] == cached_prefix:
                best_match = max(best_match, prefix_len)
        cached_tokens += best_match

    hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'hit_rate': round(hit_rate, 4),
        'cached_tokens': cached_tokens,
        'total_tokens': total_tokens
    }
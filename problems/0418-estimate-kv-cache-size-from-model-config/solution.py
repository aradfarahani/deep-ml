def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    num_layers = model_config['num_layers']
    num_attention_heads = model_config['num_attention_heads']
    num_kv_heads = model_config.get('num_kv_heads', num_attention_heads)
    hidden_size = model_config['hidden_size']
    dtype_bytes = model_config['dtype_bytes']

    head_dim = hidden_size // num_attention_heads

    # Each layer caches both K and V, each of shape [batch, num_kv_heads, seq_len, head_dim]
    kv_cache_elements = batch_size * num_layers * 2 * num_kv_heads * head_dim * seq_len

    kv_cache_size_bytes = kv_cache_elements * dtype_bytes
    kv_cache_size_mb = round(kv_cache_size_bytes / (1024 ** 2), 4)
    per_layer_size_mb = round(kv_cache_size_bytes / num_layers / (1024 ** 2), 4)

    # Per token for a single sequence (not multiplied by batch)
    per_token_elements = num_layers * 2 * num_kv_heads * head_dim
    per_token_bytes = per_token_elements * dtype_bytes
    per_token_size_kb = round(per_token_bytes / 1024, 4)

    return {
        'kv_cache_elements': kv_cache_elements,
        'kv_cache_size_bytes': kv_cache_size_bytes,
        'kv_cache_size_mb': kv_cache_size_mb,
        'per_layer_size_mb': per_layer_size_mb,
        'per_token_size_kb': per_token_size_kb
    }
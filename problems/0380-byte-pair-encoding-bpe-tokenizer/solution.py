def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
    """
    tokens = dict(corpus)
    merges = []
    
    for _ in range(num_merges):
        # Count all adjacent pairs
        pairs = {}
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        if not pairs:
            break
        
        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # Merge the best pair in all words
        new_tokens = {}
        for word, freq in tokens.items():
            symbols = word.split()
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_tokens[' '.join(new_symbols)] = freq
        
        tokens = new_tokens
    
    return merges
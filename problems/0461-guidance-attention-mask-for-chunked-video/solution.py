import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    total = sum(chunk_sizes)
    mask = np.zeros((total, total), dtype=bool)

    # compute start index of each chunk
    starts = []
    s = 0
    for size in chunk_sizes:
        starts.append(s)
        s += size

    hist_end = starts[current_chunk]

    # History tokens attend to all history tokens (bidirectional)
    mask[:hist_end, :hist_end] = True

    # Current chunk attends to all history tokens
    curr_start = starts[current_chunk]
    curr_end = curr_start + chunk_sizes[current_chunk]
    mask[curr_start:curr_end, :hist_end] = True

    # Current chunk is causal within itself
    for i in range(chunk_sizes[current_chunk]):
        mask[curr_start + i, curr_start:curr_start + i + 1] = True

    return mask
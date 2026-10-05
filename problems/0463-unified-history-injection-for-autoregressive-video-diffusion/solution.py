import numpy as np

def unified_history_injection(
    history_latents: np.ndarray,
    noisy_latents: np.ndarray,
    mode: str
) -> dict:
    T_h = history_latents.shape[0]
    T_n = noisy_latents.shape[0]

    # Representation control: modify history based on generation mode
    if mode == "t2v":
        # Text-to-video: zero out all history (no visual conditioning)
        processed_history = np.zeros_like(history_latents)
    elif mode == "i2v":
        # Image-to-video: keep only the last frame, zero the rest
        processed_history = np.zeros_like(history_latents)
        processed_history[-1] = history_latents[-1]
    elif mode == "v2v":
        # Video-to-video: keep entire history as-is
        processed_history = history_latents.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Concatenate history and noisy latents along temporal axis
    unified_input = np.concatenate([processed_history, noisy_latents], axis=0)

    # Build context mask: 0 = clean history, 1 = noisy generation target
    context_mask = np.array([0] * T_h + [1] * T_n, dtype=np.int32)

    return {
        'unified_input': unified_input,
        'context_mask': context_mask,
        'num_history_frames': T_h,
        'num_noisy_frames': T_n
    }
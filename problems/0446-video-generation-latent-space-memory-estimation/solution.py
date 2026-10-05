import math

def estimate_video_latent_memory(num_frames: int, height: int, width: int,
                                  latent_channels: int, spatial_compression: int,
                                  temporal_compression: int, batch_size: int = 1,
                                  dtype: str = "fp16", patch_size: int = 1) -> dict:
    # Compute latent dimensions using ceiling division
    latent_t = math.ceil(num_frames / temporal_compression)
    latent_h = math.ceil(height / spatial_compression)
    latent_w = math.ceil(width / spatial_compression)

    latent_shape = (batch_size, latent_channels, latent_t, latent_h, latent_w)

    # Total elements in the latent tensor
    num_elements = batch_size * latent_channels * latent_t * latent_h * latent_w

    # Bytes per element for each dtype
    bytes_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
    bpe = bytes_map[dtype]

    memory_bytes = num_elements * bpe
    memory_mb = round(memory_bytes / (1024 ** 2), 4)

    # Tokens per video after spatial patchification
    patch_h = math.ceil(latent_h / patch_size)
    patch_w = math.ceil(latent_w / patch_size)
    tokens_per_video = latent_t * patch_h * patch_w

    return {
        "latent_shape": latent_shape,
        "num_elements": num_elements,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_mb,
        "tokens_per_video": tokens_per_video
    }
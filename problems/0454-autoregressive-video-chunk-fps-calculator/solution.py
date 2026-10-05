def compute_video_generation_fps(
    num_chunks: int,
    chunk_frames: int,
    denoising_steps: int,
    time_per_step_ms: float,
    context_encoding_ms: float = 0.0,
    realtime_fps_threshold: float = 24.0
) -> dict:
    # Time to generate one chunk = denoising loop + context encoding overhead
    time_per_chunk_ms = denoising_steps * time_per_step_ms + context_encoding_ms

    # Total generation time across all chunks
    total_time_ms = num_chunks * time_per_chunk_ms
    total_time_s = round(total_time_ms / 1000, 4)

    # Total frames produced
    total_frames = num_chunks * chunk_frames

    # Frames per second
    fps = round(total_frames / total_time_s, 2)

    return {
        "total_frames": total_frames,
        "total_time_ms": total_time_ms,
        "total_time_s": total_time_s,
        "fps": fps,
        "time_per_chunk_ms": time_per_chunk_ms,
        "is_realtime": fps >= realtime_fps_threshold
    }
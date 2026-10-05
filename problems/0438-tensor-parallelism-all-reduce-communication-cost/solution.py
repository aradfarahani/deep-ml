def tensor_parallel_allreduce_cost(
    hidden_size: int,
    sequence_length: int,
    batch_size: int,
    num_gpus: int,
    num_layers: int,
    bytes_per_element: int = 2,
    bandwidth_gb_per_sec: float = 300.0,
    allreduces_per_layer: int = 2
) -> dict:
    """
    Calculate the all-reduce communication cost for tensor parallelism.
    """
    # Message size: the activation tensor being all-reduced
    message_elements = batch_size * sequence_length * hidden_size
    message_size_bytes = message_elements * bytes_per_element

    # Ring all-reduce communication volume per GPU
    # Ring all-reduce has two phases (reduce-scatter + all-gather)
    # Each phase transfers (P-1)/P of the message
    # Total per GPU: 2 * (P-1)/P * message_size
    comm_volume_per_allreduce = 2.0 * (num_gpus - 1) / num_gpus * message_size_bytes

    # Total communication volume across all layers
    total_comm_volume = comm_volume_per_allreduce * allreduces_per_layer * num_layers

    # Convert bandwidth from GB/s to bytes/s
    bandwidth_bytes_per_sec = bandwidth_gb_per_sec * 1e9

    # Total communication time in milliseconds
    if bandwidth_bytes_per_sec > 0:
        total_comm_time_ms = (total_comm_volume / bandwidth_bytes_per_sec) * 1000.0
    else:
        total_comm_time_ms = float('inf')

    return {
        'message_size_bytes': int(message_size_bytes),
        'comm_volume_per_allreduce_bytes': round(float(comm_volume_per_allreduce), 4),
        'total_comm_volume_bytes': round(float(total_comm_volume), 4),
        'total_comm_time_ms': round(float(total_comm_time_ms), 4)
    }
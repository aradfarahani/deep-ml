def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    total_params = 0
    
    for layer in layers:
        layer_type = layer["type"]
        
        if layer_type == "dense":
            input_size = layer["input_size"]
            output_size = layer["output_size"]
            use_bias = layer.get("use_bias", True)
            
            params = input_size * output_size
            if use_bias:
                params += output_size
            total_params += params
            
        elif layer_type == "conv2d":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]
            use_bias = layer.get("use_bias", True)
            
            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size
            
            params = kernel_h * kernel_w * in_channels * out_channels
            if use_bias:
                params += out_channels
            total_params += params
            
        elif layer_type == "embedding":
            num_embeddings = layer["num_embeddings"]
            embedding_dim = layer["embedding_dim"]
            params = num_embeddings * embedding_dim
            total_params += params
    
    return total_params
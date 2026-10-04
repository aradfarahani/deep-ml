def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	total_params = 0
	for layer in layers:
		layer_type = layer['type']
		has_bias = layer.get('bias', True)
		if layer_type == 'dense':
			params = layer['input_size'] * layer['output_size']
			if has_bias:
				params += layer['output_size']
		elif layer_type == 'conv2d':
			k = layer['kernel_size']
			params = layer['in_channels'] * layer['out_channels'] * k * k
			if has_bias:
				params += layer['out_channels']
		else:
			params = 0
		total_params += params
	return total_params
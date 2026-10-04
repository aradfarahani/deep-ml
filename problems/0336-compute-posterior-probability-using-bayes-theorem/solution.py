def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	# Calculate numerators: P(E|H_i) * P(H_i) for each hypothesis
	numerators = [p * l for p, l in zip(priors, likelihoods)]
	
	# Calculate total evidence probability (denominator)
	evidence = sum(numerators)
	
	# Handle edge case where evidence is zero
	if evidence == 0:
		return [0.0] * len(priors)
	
	# Calculate posterior probabilities
	posteriors = [n / evidence for n in numerators]
	
	return posteriors
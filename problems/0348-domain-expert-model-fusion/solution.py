def fuse_domain_experts(
    expert_scores: dict[str, dict[str, float]],
    domain_weights: dict[str, float],
    fusion_method: str
) -> float:
    """
    Fuse multiple domain expert models into a single score.
    """
    domains = list(domain_weights.keys())
    experts = list(expert_scores.keys())
    
    fused_domain_scores = {}
    
    if fusion_method == 'best_per_domain':
        # For each domain, take the best expert's score
        for domain in domains:
            best_score = max(expert_scores[expert].get(domain, 0) for expert in experts)
            fused_domain_scores[domain] = best_score
    
    elif fusion_method == 'weighted_average':
        # For each domain, compute weighted average based on relative performance
        for domain in domains:
            scores = [expert_scores[expert].get(domain, 0) for expert in experts]
            total = sum(scores)
            if total == 0:
                fused_domain_scores[domain] = 0
            else:
                # Weight each expert by their relative performance on this domain
                weighted_sum = sum(s * s for s in scores)  # s * (s/total) * total = s^2
                fused_domain_scores[domain] = weighted_sum / total
    
    # Compute overall score weighted by domain importance
    overall_score = sum(
        fused_domain_scores[domain] * domain_weights[domain]
        for domain in domains
    )
    
    return overall_score
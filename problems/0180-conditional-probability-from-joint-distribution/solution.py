def conditional_probability(joint_distribution: dict) -> float:
    p_ab = joint_distribution[('A','B')]
    p_notA_b = joint_distribution[('`A','B')]
    p_b = p_ab + p_notA_b
    if p_b == 0:
        raise ValueError("P(B) cannot be zero.")
    return p_ab / p_b
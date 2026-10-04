import numpy as np
import math

def analyze_ab_test(control_outcomes: list, treatment_outcomes: list, confidence_level: float = 0.95, min_detectable_effect: float = 0.02) -> dict:
    """
    Analyze A/B test results for model comparison with statistical rigor.
    
    Args:
        control_outcomes: List of binary outcomes (0 or 1) for control group
        treatment_outcomes: List of binary outcomes (0 or 1) for treatment group
        confidence_level: Confidence level for statistical tests (default 0.95)
        min_detectable_effect: Minimum absolute effect size considered practically significant
    
    Returns:
        dict with statistical analysis results and recommendation
    """
    if not control_outcomes or not treatment_outcomes:
        return {}
    
    control = np.array(control_outcomes, dtype=float)
    treatment = np.array(treatment_outcomes, dtype=float)
    
    n_c = len(control)
    n_t = len(treatment)
    
    p_c = np.mean(control)
    p_t = np.mean(treatment)
    
    absolute_lift = p_t - p_c
    
    if p_c > 0:
        relative_lift_pct = ((p_t - p_c) / p_c) * 100
    elif p_t > 0:
        relative_lift_pct = float('inf')
    else:
        relative_lift_pct = 0.0
    
    p_pooled = (np.sum(control) + np.sum(treatment)) / (n_c + n_t)
    
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_c + 1/n_t)) if p_pooled > 0 and p_pooled < 1 else 0.0
    
    z_stat = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    p_value = 2 * (1 - norm_cdf(abs(z_stat)))
    
    alpha = 1 - confidence_level
    
    def norm_ppf(p):
        if p <= 0:
            return -np.inf
        if p >= 1:
            return np.inf
        if p == 0.5:
            return 0.0
        if p > 0.5:
            return -norm_ppf(1 - p)
        t = np.sqrt(-2 * np.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
    
    z_crit = norm_ppf(1 - alpha/2)
    
    se_unpooled = np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t) if (p_c > 0 or p_t > 0) else 0.0
    
    ci_lower = absolute_lift - z_crit * se_unpooled
    ci_upper = absolute_lift + z_crit * se_unpooled
    
    statistically_significant = bool(p_value < alpha)
    practically_significant = bool(abs(absolute_lift) >= min_detectable_effect)
    
    z_alpha = z_crit
    z_beta = 0.84
    p_avg = (p_c + p_t) / 2
    
    if min_detectable_effect > 0 and p_avg > 0 and p_avg < 1:
        required_n = int(np.ceil(2 * ((z_alpha + z_beta) / min_detectable_effect)**2 * p_avg * (1 - p_avg)))
        sample_size_adequate = min(n_c, n_t) >= required_n
    else:
        required_n = 0
        sample_size_adequate = True
    
    if statistically_significant and practically_significant and absolute_lift > 0:
        recommendation = 'launch_treatment'
    elif statistically_significant and (absolute_lift < 0 or not practically_significant):
        recommendation = 'keep_control'
    else:
        recommendation = 'continue_testing'
    
    return {
        'control_rate': round(float(p_c), 4),
        'treatment_rate': round(float(p_t), 4),
        'absolute_lift': round(float(absolute_lift), 4),
        'relative_lift_pct': round(float(relative_lift_pct), 2) if relative_lift_pct != float('inf') else 'inf',
        'z_statistic': round(float(z_stat), 4),
        'p_value': round(float(p_value), 4),
        'confidence_interval': (round(float(ci_lower), 4), round(float(ci_upper), 4)),
        'statistically_significant': statistically_significant,
        'practically_significant': practically_significant,
        'required_sample_size': required_n,
        'recommendation': recommendation
    }
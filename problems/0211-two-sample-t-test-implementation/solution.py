import numpy as np
import math

def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).
    """
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    n1, n2 = len(s1), len(s2)

    # Means and unbiased variances (Bessel's correction)
    mean1, mean2 = np.mean(s1), np.mean(s2)
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)

    # Welch standard error and t-statistic
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / se

    # WelchâSatterthwaite degrees of freedom (fractional)
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )

    # Two-tailed p-value via regularized incomplete beta:
    #   P(|T| > |t|) = I_x(df/2, 1/2)   with   x = df / (df + t^2)
    t_abs = abs(t_stat)
    x = df / (df + t_abs * t_abs)
    p_value = _betainc(df / 2.0, 0.5, x)

    reject_null = p_value < alpha

    # Cohen's d using pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': float(df),
        'reject_null': bool(reject_null),
        'cohens_d': float(cohens_d),
    }
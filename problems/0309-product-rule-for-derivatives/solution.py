import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    f = np.array(f_coeffs, dtype=float)
    g = np.array(g_coeffs, dtype=float)
    
    # Compute derivative of f: d/dx[a0 + a1*x + a2*x^2 + ...] = a1 + 2*a2*x + ...
    if len(f) <= 1:
        f_prime = np.array([0.0])
    else:
        f_prime = f[1:] * np.arange(1, len(f))
    
    # Compute derivative of g
    if len(g) <= 1:
        g_prime = np.array([0.0])
    else:
        g_prime = g[1:] * np.arange(1, len(g))
    
    # Apply product rule: (f*g)' = f'*g + f*g'
    term1 = np.convolve(f_prime, g)
    term2 = np.convolve(f, g_prime)
    
    # Pad to same length and add
    max_len = max(len(term1), len(term2))
    result = np.zeros(max_len)
    result[:len(term1)] += term1
    result[:len(term2)] += term2
    
    # Remove trailing zeros but keep at least one element
    while len(result) > 1 and abs(result[-1]) < 1e-10:
        result = result[:-1]
    
    return [round(x, 4) for x in result.tolist()]
import numpy as np

def compare_formats(values):
    values = np.array(values, dtype=np.float64)
    
    format_specs = {
        'fp16':      {'exp_bits': 5,  'man_bits': 10, 'has_inf': True},
        'bf16':      {'exp_bits': 8,  'man_bits': 7,  'has_inf': True},
        'fp8_e4m3':  {'exp_bits': 4,  'man_bits': 3,  'has_inf': False},
        'fp4_e2m1':  {'exp_bits': 2,  'man_bits': 1,  'has_inf': False},
    }
    
    def get_format_properties(exp_bits, man_bits, has_inf):
        bias = (1 << (exp_bits - 1)) - 1
        if has_inf:
            max_biased_exp = (1 << exp_bits) - 2
            max_mantissa_frac = ((1 << man_bits) - 1) / (1 << man_bits)
        else:
            max_biased_exp = (1 << exp_bits) - 1
            max_mantissa_frac = ((1 << man_bits) - 2) / (1 << man_bits)
        max_exp = max_biased_exp - bias
        max_val = (2.0 ** max_exp) * (1.0 + max_mantissa_frac)
        min_exp = 1 - bias
        min_pos_normal = 2.0 ** min_exp
        return bias, max_exp, min_exp, max_val, min_pos_normal
    
    def quantize_value(v, exp_bits, man_bits, has_inf):
        if v == 0.0 or (np.isnan(v)):
            return 0.0
        
        bias, max_exp, min_exp, max_val, min_pos_normal = get_format_properties(exp_bits, man_bits, has_inf)
        
        sign = 1.0 if v >= 0 else -1.0
        abs_val = abs(v)
        
        if abs_val > max_val:
            if has_inf:
                return sign * float('inf')
            else:
                return sign * max_val
        
        log2_val = np.floor(np.log2(abs_val))
        exp = int(log2_val)
        
        if exp < min_exp:
            step = 2.0 ** (min_exp - man_bits)
            quantized = np.round(abs_val / step) * step
            if quantized >= min_pos_normal:
                quantized = min_pos_normal
            if quantized == 0.0:
                return 0.0
            return sign * float(quantized)
        
        if exp > max_exp:
            if has_inf:
                return sign * float('inf')
            else:
                return sign * max_val
        
        step = 2.0 ** (exp - man_bits)
        quantized = np.round(abs_val / step) * step
        
        if quantized >= 2.0 ** (exp + 1):
            exp += 1
            if exp > max_exp:
                if has_inf:
                    return sign * float('inf')
                else:
                    return sign * max_val
            step = 2.0 ** (exp - man_bits)
            quantized = np.round(abs_val / step) * step
        
        return sign * float(quantized)
    
    results = {}
    for name, spec in format_specs.items():
        bias, max_exp, min_exp, max_val, min_pos_normal = get_format_properties(**spec)
        
        quantized = [quantize_value(float(v), **spec) for v in values]
        
        abs_errors = []
        for q, v in zip(quantized, values):
            if np.isinf(q) and not np.isinf(v):
                abs_errors.append(float('inf'))
            else:
                abs_errors.append(abs(q - float(v)))
        
        max_abs_err = max(abs_errors) if abs_errors else 0.0
        mean_abs_err = np.mean(abs_errors) if abs_errors else 0.0
        
        if not np.isinf(max_abs_err):
            max_abs_err = round(float(max_abs_err), 6)
        if not np.isinf(mean_abs_err):
            mean_abs_err = round(float(mean_abs_err), 6)
        
        results[name] = {
            'max_representable': float(max_val),
            'min_positive_normal': float(min_pos_normal),
            'quantized': [float(q) for q in quantized],
            'max_abs_error': max_abs_err,
            'mean_abs_error': mean_abs_err,
        }
    
    return results
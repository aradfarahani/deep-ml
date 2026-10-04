import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    """
    if not test_cases:
        return {
            'pass_rate': 0.0,
            'error_rate': 0.0,
            'passed_count': 0,
            'total_count': 0,
            'verdicts': []
        }
    
    verdicts = []
    passed_count = 0
    error_count = 0
    
    for test in test_cases:
        status = test['status']
        expected = test['expected']
        actual = test.get('actual')
        
        if status != 'success':
            verdicts.append('error')
            error_count += 1
            continue
        
        if actual is None:
            verdicts.append('fail')
            continue
        
        # Normalize: strip whitespace
        expected_norm = expected.strip()
        actual_norm = actual.strip()
        
        # Try numeric comparison first
        try:
            exp_val = float(expected_norm)
            act_val = float(actual_norm)
            if abs(exp_val - act_val) <= numeric_tolerance:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')
        except (ValueError, TypeError):
            # String comparison
            if expected_norm == actual_norm:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')
    
    total = len(test_cases)
    
    return {
        'pass_rate': round(passed_count / total, 4),
        'error_rate': round(error_count / total, 4),
        'passed_count': passed_count,
        'total_count': total,
        'verdicts': verdicts
    }
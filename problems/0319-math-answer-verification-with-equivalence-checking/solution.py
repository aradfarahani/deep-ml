import re
import math

def verify_math_answer(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Verify if two mathematical answers are equivalent.
    """
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()
    
    # Direct string equality check
    if predicted == ground_truth:
        return True
    
    # Try to evaluate both to numerical values
    pred_val = _parse_math_value(predicted)
    truth_val = _parse_math_value(ground_truth)
    
    if pred_val is not None and truth_val is not None:
        return abs(pred_val - truth_val) <= tolerance
    
    return False

def _parse_math_value(expr: str):
    """Parse a mathematical expression to a float value."""
    if not expr:
        return None
    
    expr = expr.strip()
    
    # Try direct float conversion
    try:
        return float(expr)
    except ValueError:
        pass
    
    # Replace 'pi' with numerical value
    expr = re.sub(r'\bpi\b', str(math.pi), expr)
    
    # Replace sqrt(x) with computed value
    while 'sqrt(' in expr:
        match = re.search(r'sqrt\(([^()]+)\)', expr)
        if match:
            inner = match.group(1)
            try:
                inner_val = eval(inner, {"__builtins__": {}})
                inner_val = float(inner_val)
                if inner_val >= 0:
                    expr = expr[:match.start()] + str(math.sqrt(inner_val)) + expr[match.end():]
                else:
                    return None
            except:
                return None
        else:
            break
    
    # Final evaluation
    try:
        result = eval(expr, {"__builtins__": {}})
        return float(result)
    except:
        return None
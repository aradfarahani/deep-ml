import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives at a point.
    
    Args:
        func_name: Function name
        point: Point at which to evaluate derivatives
    
    Returns:
        Tuple of partial derivatives
    """
    if func_name == 'poly2d':
        # f(x,y) = xÂ²y + xyÂ²
        x, y = point
        df_dx = 2*x*y + y**2
        df_dy = x**2 + 2*x*y
        return (float(df_dx), float(df_dy))
    
    elif func_name == 'exp_sum':
        # f(x,y) = e^(x+y)
        x, y = point
        exp_val = np.exp(x + y)
        return (float(exp_val), float(exp_val))
    
    elif func_name == 'product_sin':
        # f(x,y) = x*sin(y)
        x, y = point
        df_dx = np.sin(y)
        df_dy = x * np.cos(y)
        return (float(df_dx), float(df_dy))
    
    elif func_name == 'poly3d':
        # f(x,y,z) = xÂ²y + yzÂ²
        x, y, z = point
        df_dx = 2*x*y
        df_dy = x**2 + z**2
        df_dz = 2*y*z
        return (float(df_dx), float(df_dy), float(df_dz))
    
    elif func_name == 'squared_error':
        # f(x,y) = (x-y)Â²
        x, y = point
        df_dx = 2*(x - y)
        df_dy = -2*(x - y)
        return (float(df_dx), float(df_dy))
import numpy as np

T_TABLE = {
    (4, 0.95):  2.1318467863998393,
    (4, 0.975): 2.7764451051977987,
    (4, 0.995): 4.604094871415387,
    (5, 0.95):  2.0150483726691575,
    (5, 0.975): 2.5705818366147395,
    (5, 0.995): 4.032142983557536,
    (6, 0.95):  1.9431802803927816,
    (6, 0.975): 2.4469118511449624,
    (6, 0.995): 3.707428021324907,
    (7, 0.95):  1.8945786050613064,
    (7, 0.975): 2.3646242510102993,
    (7, 0.995): 3.4994832973505026,
}


def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    data = np.asarray(data, dtype=float)
    n = len(data)

    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    standard_error = sample_std / np.sqrt(n)

    alpha = 1 - confidence_level
    df = n - 1
    t_critical = T_TABLE[(df, round(1 - alpha / 2, 3))]

    margin_of_error = t_critical * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return {
        'mean': float(sample_mean),
        'standard_error': float(standard_error),
        'margin_of_error': float(margin_of_error),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'confidence_level': float(confidence_level),
    }
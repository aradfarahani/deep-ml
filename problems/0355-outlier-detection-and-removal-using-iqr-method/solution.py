import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	arr = np.array(data, dtype=float)
	q1 = np.percentile(arr, 25)
	q3 = np.percentile(arr, 75)
	iqr = q3 - q1
	lower_bound = q1 - k * iqr
	upper_bound = q3 + k * iqr
	
	outlier_mask = (arr < lower_bound) | (arr > upper_bound)
	outlier_indices = np.where(outlier_mask)[0].tolist()
	cleaned_data = arr[~outlier_mask].tolist()
	
	return {
		'cleaned_data': [round(float(x), 4) for x in cleaned_data],
		'outlier_indices': outlier_indices,
		'lower_bound': round(float(lower_bound), 4),
		'upper_bound': round(float(upper_bound), 4)
	}
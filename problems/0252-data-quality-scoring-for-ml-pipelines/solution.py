def calculate_data_quality_score(data: list, schema: dict) -> dict:
    """
    Calculate data quality metrics for ML pipeline monitoring.
    
    Args:
        data: list of dictionaries representing rows of data
        schema: dictionary defining expected columns and their types
                {'column_name': {'type': 'numeric'|'categorical'|'boolean', 'nullable': True|False}}
    
    Returns:
        dict with keys: 'completeness', 'type_validity', 'uniqueness_ratio', 'overall_score'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not data:
        return {}
    
    total_fields = 0
    non_null_fields = 0
    valid_type_fields = 0
    
    for row in data:
        for col, spec in schema.items():
            total_fields += 1
            value = row.get(col)
            
            if value is not None:
                non_null_fields += 1
                
                expected_type = spec.get('type')
                if expected_type == 'numeric':
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        valid_type_fields += 1
                elif expected_type == 'categorical':
                    if isinstance(value, str):
                        valid_type_fields += 1
                elif expected_type == 'boolean':
                    if isinstance(value, bool):
                        valid_type_fields += 1
            else:
                if spec.get('nullable', True):
                    valid_type_fields += 1
    
    completeness = (non_null_fields / total_fields) * 100 if total_fields > 0 else 0
    type_validity = (valid_type_fields / total_fields) * 100 if total_fields > 0 else 0
    
    row_tuples = [tuple(sorted(row.items())) for row in data]
    uniqueness_ratio = (len(set(row_tuples)) / len(data)) * 100
    
    overall_score = 0.4 * completeness + 0.4 * type_validity + 0.2 * uniqueness_ratio
    
    return {
        'completeness': round(completeness, 2),
        'type_validity': round(type_validity, 2),
        'uniqueness_ratio': round(uniqueness_ratio, 2),
        'overall_score': round(overall_score, 2)
    }
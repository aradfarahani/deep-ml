def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    if mode == 'row':
        means = [sum(row) / len(row) for row in matrix]
    elif mode == 'column':
        num_cols = len(matrix[0])
        means = [sum(matrix[row][col] for row in range(len(matrix))) / len(matrix) for col in range(num_cols)]
    else:
        raise ValueError("Mode must be 'row' or 'column'")
    
    return means

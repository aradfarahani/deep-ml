def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(x) for x in zip(*a)]
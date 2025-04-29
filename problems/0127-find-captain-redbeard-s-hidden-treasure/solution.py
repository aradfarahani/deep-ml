def find_treasure(start_x: float) -> float:
    learning_rate = 0.01
    tolerance = 1e-6
    max_iters = 10000

    def gradient(x):
        return 4 * x**3 - 9 * x**2

    def f(x):
        return x**4 - 3*x**3 + 2

    def descend(x0):
        x = x0
        for _ in range(max_iters):
            grad = gradient(x)
            new_x = x - learning_rate * grad
            if abs(new_x - x) < tolerance:
                break
            x = new_x
        return x

    # Try multiple initial points
    candidates = [start_x, 0, 1, 2, 3, -1, -2]
    best_x = min((descend(x) for x in candidates), key=f)
    return round(best_x, 4)

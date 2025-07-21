def conditional_probability(data, x, y):
    # Filter data for all instances where X = x
    filtered = [item for item in data if item[0] == x]
    if not filtered:
        return 0.0
    count_y = sum(1 for item in filtered if item[1] == y)
    return round(count_y / len(filtered), 4)
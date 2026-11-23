def parse_points(points_str: str) -> list[tuple[float, float]]:
    points = []
    for point in points_str.split():
        x, y = point.split(",")
        points.append((float(x), float(y)))
    return points


def filter_objects_by_color(objects, color):
    """Filter objects list by a specific color."""
    return [obj for obj in objects if obj['color'] == color]

def update_objects_color(grid, objects, new_color):
    """Update the color of specified objects in the grid. Returns NEW grid."""
    new_grid = [row[:] for row in grid]
    for obj in objects:
        for r, c in obj['pixels']:
            new_grid[r][c] = new_color
    return new_grid

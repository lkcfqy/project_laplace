# src/dsl.py
"""
Domain Specific Language (DSL) Module
------------------------------------
This module defines the primitives available to the agent for solving ARC tasks.
It includes functions for object detection, grid manipulation, and color counting.
"""
import copy

# --- 核心工具函数 ---

def get_objects(grid, bg_color=0):
    """
    物体检测：返回所有非背景色的连通区域。
    返回格式: [{'color': int, 'pixels': [(r,c), ...]}, ...]
    """
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    objects = []

    def dfs(r, c, color, current_obj):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != color:
            return
        visited.add((r, c))
        current_obj.append((r, c))
        # 4-neighbor connectivity
        dfs(r+1, c, color, current_obj)
        dfs(r-1, c, color, current_obj)
        dfs(r, c+1, color, current_obj)
        dfs(r, c-1, color, current_obj)

    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color != bg_color and (r, c) not in visited:
                current_obj = []
                dfs(r, c, color, current_obj)
                objects.append({'color': color, 'pixels': current_obj})
    return objects

def crop_grid(grid, top, left, height, width):
    """裁剪网格"""
    return [row[left:left+width] for row in grid[top:top+height]]

def flood_fill(grid, r, c, target_color):
    """泛洪填充：将 (r,c) 所在的连通区域染成 target_color"""
    new_grid = copy.deepcopy(grid)
    rows, cols = len(new_grid), len(new_grid[0])
    start_color = new_grid[r][c]
    if start_color == target_color:
        return new_grid
    
    stack = [(r, c)]
    visited = set()
    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in visited: continue
        visited.add((curr_r, curr_c))
        
        if new_grid[curr_r][curr_c] == start_color:
            new_grid[curr_r][curr_c] = target_color
            if curr_r > 0: stack.append((curr_r-1, curr_c))
            if curr_r < rows-1: stack.append((curr_r+1, curr_c))
            if curr_c > 0: stack.append((curr_r, curr_c-1))
            if curr_c < cols-1: stack.append((curr_r, curr_c+1))
    return new_grid

def count_colors(grid):
    """统计颜色数量，返回字典 {color: count}"""
    counts = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts

    return counts

def rotate_90(grid, k=1):
    """Rotate grid 90 degrees clockwise k times."""
    result = grid
    for _ in range(k % 4):
        result = [list(row) for row in zip(*result[::-1])]
    return result

def flip_vertical(grid):
    """Flip grid vertically."""
    return grid[::-1]

def flip_horizontal(grid):
    """Flip grid horizontally."""
    return [row[::-1] for row in grid]

def move_object(grid, obj, dr, dc, bg_color=0):
    """Move an object by (dr, dc) offset. Returns NEW grid."""
    new_grid = [row[:] for row in grid]
    # First clear the object from its old position
    for r, c in obj['pixels']:
        new_grid[r][c] = bg_color
    
    # Draw at new position
    for r, c in obj['pixels']:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            new_grid[nr][nc] = obj['color']
    return new_grid

def merge_objects(dims, objects, bg_color=0):
    """Create a new grid from a list of objects."""
    rows, cols = dims
    grid = [[bg_color for _ in range(cols)] for _ in range(rows)]
    for obj in objects:
        for r, c in obj['pixels']:
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = obj['color']
    return grid

def roll_grid(grid, shift_r=0, shift_c=0):
    """Roll (shift cyclically) grid rows and columns."""
    rows = len(grid)
    cols = len(grid[0])
    # Shift rows
    new_grid = grid[-shift_r % rows:] + grid[:-shift_r % rows]
    # Shift cols
    result = []
    for r in range(rows):
        row = new_grid[r]
        result.append(row[-shift_c % cols:] + row[:-shift_c % cols])
    return result

def scale_object(obj, scale_factor):
    """Scale an object key's pixels. Crude nearest neighbor."""
    if scale_factor == 1: return obj
    new_pixels = []
    # Find center or simple bounding box expansion
    # Implementation simplified: expand each pixel to a block
    for r, c in obj['pixels']:
        for dr in range(scale_factor):
            for dc in range(scale_factor):
                new_pixels.append((r * scale_factor + dr, c * scale_factor + dc))
    return {'color': obj['color'], 'pixels': new_pixels}

def place_object(grid, obj, r_offset, c_offset):
    """Place an object onto grid at offset. Overwrites existing pixels."""
    new_grid = [row[:] for row in grid]
    rows, cols = len(new_grid), len(new_grid[0])
    for r, c in obj['pixels']:
        nr, nc = r + r_offset, c + c_offset
        if 0 <= nr < rows and 0 <= nc < cols:
            new_grid[nr][nc] = obj['color']
    return new_grid

def detect_periodicity(grid):
    """Detect if grid is tiling of a smaller patch. Returns (h, w) of tile or None."""
    rows, cols = len(grid), len(grid[0])
    for h in range(1, rows // 2 + 1):
        if rows % h == 0:
            # Check vertical tiling
            is_v_tile = True
            base = grid[:h]
            for k in range(1, rows // h):
                if grid[k*h:(k+1)*h] != base:
                    is_v_tile = False
                    break
            if is_v_tile:
                # Vertical Periodicity Found, check horizontal
                # Detailed impl omitted for brevity, returning simple vertical period
                return (h, cols) # Means rows repeat every h
    return None

def evolve(grid, iterations=1):
    """Conway's Game of Life style evolution. 
    Rule: Simple majority or custom rule could be passed. 
    Here implementing a 'color spread' rule commonly found in ARC:
    - If a cell is surrounded by >4 neighbors of color X, it becomes X.
    """
    current = grid
    rows, cols = len(grid), len(grid[0])
    
    for _ in range(iterations):
        nxt = [row[:] for row in current]
        for r in range(rows):
            for c in range(cols):
                # Count neighbors
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(current[nr][nc])
                
                # Apply simple rule: Adoption
                if neighbors:
                    counts = {}
                    for n in neighbors: counts[n] = counts.get(n, 0) + 1
                    most_common = max(counts.items(), key=lambda x: x[1])
                    if most_common[1] >= 5: # Threshold
                        nxt[r][c] = most_common[0]
        current = nxt
    return current

# --- New Helper Functions ---

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

# --- Prompt 提示词 ---
# 这段文字会被塞进 System Prompt，告诉 LLM 它有哪些武器可用
DSL_PROMPT = """
[Helper Library Available]
You can use the following pre-defined functions to simplify your code (NO need to implement them):

1. `get_objects(grid, bg_color=0)`
   - Returns a list of objects: [{'color': int, 'pixels': [(r,c),...]}, ...]
   - Useful for moving, counting, or filtering shapes.

2. `crop_grid(grid, top, left, height, width)`
   - Returns a sub-grid (2D list).

3. `flood_fill(grid, r, c, target_color)`
   - Returns a NEW grid with the connected area filled.

4. `count_colors(grid)`
   - Returns a dict {color: count}.

98. `rotate_90(grid, k=1)` / `flip_vertical(grid)` / `flip_horizontal(grid)`
99. `move_object(grid, obj, dr, dc, bg_color=0)`
    - Returns NEW grid with object moved.
101. `roll_grid(grid, shift_r, shift_c)`
102. `scale_object(obj, factor)` / `place_object(grid, obj, r, c)`
103. `detect_periodicity(grid)` -> (h, w)
    - Returns tile size if periodic.
104. `evolve(grid, iterations=1)`
    - Runs cellular automata rules (neighborhood voting).
105. `merge_objects(dims, objects, bg_color=0)`
    - Reconstructs grid from objects.

106. `filter_objects_by_color(objects, color)`
    - Returns list of objects with specific color.
107. `update_objects_color(grid, objects, new_color)`
    - Returns NEW grid with objects changed to new_color.

Example Usage:
   objs = get_objects(grid)
   # Move first red object right by 2
   for obj in objs:
       if obj['color'] == 1:
           grid = move_object(grid, obj, 0, 2)
"""

import json
import random
import os
import argparse
from mcts import MCTSSolver
from agent_lora import UnslothAgent
from executor import LocalSandbox
import dsl
import inspect
import config

def generate_identity_task():
    """Task: Input = Output"""
    w = random.randint(3, 6)
    h = random.randint(3, 6)
    grid = [[random.randint(0, 9) for _ in range(w)] for _ in range(h)]
    return {
        "train": [{"input": grid, "output": grid}, {"input": grid, "output": grid}],
        "test": [{"input": grid, "output": grid}],
        "description": "Output is same as input."
    }

def generate_color_swap_task():
    """Task: Swap Color A with Color B"""
    c1 = random.randint(1, 4)
    c2 = random.randint(5, 9)
    w = random.randint(3, 6)
    h = random.randint(3, 6)
    
    def make_grid():
        grid = [[random.choice([0, c1]) for _ in range(w)] for _ in range(h)]
        out = [[c2 if x == c1 else x for x in row] for row in grid]
        return grid, out
        
    g1, o1 = make_grid()
    g2, o2 = make_grid()
    gt, ot = make_grid()
    
    return {
        "train": [{"input": g1, "output": o1}, {"input": g2, "output": o2}],
        "test": [{"input": gt, "output": ot}],
        "description": f"Change color {c1} to {c2}."
    }

def generate_move_task():
    """Task: Move all non-black pixels down by 1"""
    w = random.randint(3, 6)
    h = random.randint(4, 6)
    
    def make_grid():
        # Sparse grid to avoid overlap issues for this simple test
        grid = [[0 for _ in range(w)] for _ in range(h)]
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h-2): # Leave bottom empty
            for c in range(w):
                if random.random() < 0.3:
                    color = random.randint(1, 9)
                    grid[r][c] = color
                    out[r+1][c] = color
        return grid, out
        
    g1, o1 = make_grid()
    g2, o2 = make_grid()
    gt, ot = make_grid()
    
    return {
        "train": [{"input": g1, "output": o1}, {"input": g2, "output": o2}],
        "test": [{"input": gt, "output": ot}],
        "description": "Move everything down by 1."
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of tasks to generate")
    args = parser.parse_args()
    
    print(f"🏭 Generating {args.count} synthetic tasks for data collection...")
    
    # Init Solver Components
    try:
        agent = UnslothAgent() # Load once
    except:
        print("Failed to load agent")
        return
        
    sandbox = LocalSandbox(timeout=2)
    dsl_source = inspect.getsource(dsl)
    
    solver = MCTSSolver(agent, sandbox, dsl_source)
    
    success_count = 0
    
    for i in range(args.count):
        print(f"\n--- Task {i+1}/{args.count} ---")
        
        # Mix task types
        match i % 3:
            case 0:
                task = generate_identity_task()
                print("Type: Identity")
            case 1:
                task = generate_color_swap_task()
                print("Type: Color Swap")
            case 2:
                task = generate_move_task()
                print("Type: Global Movement (Gravity)")
            
        # Run Solver (Low iterations ok for simple tasks)
        code = solver.solve(task, max_iterations=10)
        
        if code:
            success_count += 1
            
    print(f"\n✅ Data Generation Complete. Solved {success_count}/{args.count} tasks.")
    print(f"Training data saved to {config.HDC_DATA_FILE}")

if __name__ == "__main__":
    main()

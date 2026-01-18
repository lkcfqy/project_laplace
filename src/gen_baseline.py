import json
import random
import os
import dsl
import inspect

def generate_baseline_data(num_samples=2000):
    """
    Generates synthetic training data by executing DSL primitives.
    Approach: Reverse Engineering.
    1. Generate Random Grid
    2. Apply Random DSL Function (e.g. rotate, count_color, roll)
    3. Input = Grid, Output = Result
    4. Task = "Write code to transform Input to Output"
    5. Solution = The function call
    """
    print(f"🏭 Generating {num_samples} synthetic baseline samples...")
    
    data = []
    
    # Primitives to teach
    primitives = [
        ('rotate_90', lambda g: dsl.rotate_90(g)),
        ('flip_vertical', lambda g: dsl.flip_vertical(g)),
        ('flip_horizontal', lambda g: dsl.flip_horizontal(g)), 
        ('count_colors', lambda g: dsl.count_colors(g)),
        ('roll_grid_down', lambda g: dsl.roll_grid(g, shift_r=1)),
        ('roll_grid_right', lambda g: dsl.roll_grid(g, shift_c=1)),
    ]

    for _ in range(num_samples):
        # 1. Random Grid
        rows = random.randint(3, 10)
        cols = random.randint(3, 10)
        grid = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
        
        # --- Level 2: Composition (2 steps) ---
        # Pick 2 primitives that are compatible
        # We perform the operations in Python to get the ground truth
        
        # Step 1
        p1_name, p1_func = random.choice(primitives)
        try:
            grid_step1 = p1_func(grid)
            if not grid_step1: continue # Some ops might fail or return None
        except: continue
            
        # Step 2
        p2_name, p2_func = random.choice(primitives)
        try:
            grid_final = p2_func(grid_step1)
            if not grid_final: continue
        except: continue
        
        # Construct Prompt
        # Prompt needs to describe the *intent* or just ask to solve input->output
        # To teach reasoning, we might want to describe the steps in the prompt?
        # No, ARC is "Input -> Output" by example. We should give I/O and expect code.
        # But for synthetic data, we can optionally provide a "Hint" in the user prompt 
        # to make it easier for the model to map text -> code.
        description = f"Transform the grid by applying `{p1_name}` then `{p2_name}`."
        
        # Format Code
        # We need to compose the string
        # e.g. x = func1(grid); return func2(x)
        
        def format_call(func_name, arg_str):
            if func_name == 'roll_grid_down': return f"roll_grid({arg_str}, shift_r=1)"
            if func_name == 'roll_grid_right': return f"roll_grid({arg_str}, shift_c=1)"
            return f"{func_name}({arg_str})"

        code_step1 = format_call(p1_name, "grid")
        code_step2 = format_call(p2_name, "x")
        
        full_code = f"def solve(grid):\n    x = {code_step1}\n    return {code_step2}"
        
        # Occasional Level 1 (Mixed Curriculum) - 30% chance
        if random.random() < 0.3:
             full_code = f"def solve(grid):\n    return {code_step1}"
             grid_final = grid_step1
             description = f"Apply `{p1_name}` to the grid."

        # 4. JSONL Entry
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert in abstract reasoning.\n{dsl.DSL_PROMPT}"
                },
                {
                    "role": "user",
                    "content": f"Problem: {description}\n\nInput Grid:\n{grid}\n\nWrite the solution code."
                },
                {
                    "role": "assistant",
                    "content": f"```python\n{full_code}\n```"
                }
            ]
        }
        data.append(entry)

    # Save
    import config
    # Ensure config path matches or hardcode if simple
    # We use config.DSL_DATA_FILE which is synthetic_dsl_train.jsonl
    out_path = config.DSL_DATA_FILE
    
    with open(out_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Saved baseline data to {out_path}")

if __name__ == "__main__":
    generate_baseline_data()

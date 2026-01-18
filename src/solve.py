# src/solve.py
"""
Main Solver Module
-----------------
This is the entry point for solving ARC tasks. It supports multiple strategies:
1. Sampling (Best of N)
2. Refinement (Iterative Fixing)
3. MCTS (Tree Search)
"""
import argparse
import ast
import inspect
import json
import os
import re
import sys
import torch
import random
import glob

from agent_lora import UnslothAgent
from executor import LocalSandbox
import dsl
from mcts import MCTSSolver
from hdc import HDCSpace

# --- Configuration ---
TARGET_TASKS = ["25d8a9c8", "0ca9ddb6", "d037b0a7"]
import config
import datetime

# --- Helper Functions ---

def load_arc_task(task_path):
    with open(task_path, 'r') as f:
        data = json.load(f)
    return data

def save_successful_trace(task_id, code, task_data):
    """Save the successful code as a training example (Dream)."""
    # Construct conversation format
    # Using the same prompt format as training
    system_prompt, user_prompt = construct_prompt(task_data, inspect.getsource(dsl)) # Recalculate prompts
    
    # The assistant response is the code
    assistant_response = f"```python\n{code}\n```"
    
    entry = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ],
        "task_id": task_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "dream_solver"
    }
    
    with open(config.DREAM_DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"   💾 Saved successful trace to {config.DREAM_DATA_FILE}")

def parse_code(response_text):
    """Clean and extract code from LLM response."""
    # 1. Strip <|im_xxx|> tokens just in case
    content = response_text
    if "<|im_start|>assistant" in content:
        content = content.split("<|im_start|>assistant\n")[-1]
    content = content.replace("<|im_end|>", "").strip()
    
    # 2. Remove <think> blocks if present (DeepSeek style or similar CoT)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    
    # 3. Regex match for markdown code blocks
    # Try fully enclosed python blocks first
    code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
        
    # Try generic code blocks
    code_match = re.search(r"```(.*?)```", content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # 4. Fallback: Def match (dangerous if text is mixed, but necessary if markdown missing)
    if "def solve(" in content:
        start_idx = content.find("def solve(")
        # Attempt to capture until end of string or next double newline sequence that looks like text
        # Simple heuristic: take everything naturally
        return content[start_idx:].strip()
        
    return ""

def construct_prompt(task_data, dsl_source):
    examples_str = ""
    for i, pair in enumerate(task_data['train']):
        examples_str += f"Example {i+1}:\nInput: {pair['input']}\nOutput: {pair['output']}\n\n"
    
    system_prompt = f"""You are an expert in abstract reasoning using a DSL.
The input and output are 2D grids.
{dsl.DSL_PROMPT}

Your Goal: Write a Python function `solve(grid)` that transforms the input to output."""

    # 关键修改：如果任务包含描述（Debug模式），直接告诉模型要做什么
    # 这模拟了训练时的分布（Problem: ...）
    if "description" in task_data:
        user_prompt = f"""Problem: {task_data['description']}
        
Here are the examples for verification:
{examples_str}

Step 1: Implement the `solve(grid)` function in Python using the DSL based on the Problem description.
Return the code inside a ```python``` block."""
    else:
        # 原有的推理模式（让模型自己猜）
        user_prompt = f"""Here are the examples:
{examples_str}

Step 1: Analyze the examples to find the transformation rule.
Step 2: Implement the `solve(grid)` function in Python using the DSL.
Return the code inside a ```python``` block."""

    return system_prompt, user_prompt

def validate_candidate(sandbox, full_code, train_pairs):
    for i, pair in enumerate(train_pairs):
        inp = str(pair['input'])
        expected_out = pair['output']
        
        success, output, error = sandbox.execute(full_code, test_input=inp)
        
        if not success:
            # 截断过长的错误信息，防止爆 Context
            error_trunc = error[:1000] + ("..." if len(error) > 1000 else "")
            return False, f"Runtime Error on Example {i+1}:\n{error_trunc}"
        
        try:
            pred = ast.literal_eval(output.strip())
            if pred != expected_out:
                pred_str = str(pred)[:200]
                return False, f"Logic Error on Example {i+1}: Got {pred_str}..."
        except Exception as e:
            msg = str(e)
            return False, f"Output Parsing Error on Example {i+1}: {msg[:500]}"
            
    return True, "All Tests Passed"

# --- Solving Strategies ---

def solve_with_sampling(agent, sandbox, task_data, dsl_source, num_samples=5):
    """Strategy 1: Parallel Sampling (Best of N)"""
    print(f"   🎲 Strategy: Sampling ({num_samples} attempts)...")
    system_prompt, user_prompt = construct_prompt(task_data, dsl_source)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    inputs = agent.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    
    passed_code = None
    
    # Generate N samples (Sequential generation to save VRAM, conceptually parallel)
    for i in range(num_samples):
        torch.cuda.empty_cache()
        print(f"      Sample {i+1}/{num_samples}...")
        
        # High temperature for diversity
        generated_code = agent.generate_code(messages, temperature=0.7, max_new_tokens=1500)
        
        # Patch code if needed
        if generated_code and "def solve(" not in generated_code:
             lines = generated_code.splitlines()
             indented = ["    " + line for line in lines]
             generated_code = "def solve(grid):\n" + "\n".join(indented)

        if not generated_code: continue

        full_code = dsl_source + "\n\n" + generated_code
        passed, feedback = validate_candidate(sandbox, full_code, task_data['train'])
        
        if passed:
            print(f"   ✨ Sample {i+1} Passed Training Set!")
            passed_code = full_code
            break
            
    return passed_code

def solve_with_refinement(agent, sandbox, task_data, dsl_source, max_steps=3):
    """Strategy 2: Self-Refinement (Iterative Fixing)"""
    print(f"   🔧 Strategy: Refinement (Max {max_steps} steps)...")
    system_prompt, user_prompt = construct_prompt(task_data, dsl_source)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for step in range(max_steps):
        torch.cuda.empty_cache()
        print(f"      Step {step+1}/{max_steps}...")
        
        # Lower temperature for precision if refining, but initial generation can be higher
        temp = 0.7 if step == 0 else 0.4
        generated_code = agent.generate_code(messages, temperature=temp, max_new_tokens=1500)
        
        if generated_code and "def solve(" not in generated_code:
             lines = generated_code.splitlines()
             indented = ["    " + line for line in lines]
             generated_code = "def solve(grid):\n" + "\n".join(indented)
             
        full_code = dsl_source + "\n\n" + generated_code
        passed, feedback = validate_candidate(sandbox, full_code, task_data['train'])
        
        if passed:
            print(f"   ✨ Step {step+1} Passed Training Set!")
            return full_code
        
        # Add feedback to history
        print(f"      ❌ Failed. Feedback: {feedback[:100]}...")
        messages.append({"role": "assistant", "content": f"```python\n{generated_code}\n```"})
        messages.append({
            "role": "user", 
            "content": f"Your code failed on the training examples.\nFeedback: {feedback}\nPlease analyze why it failed and rewrite the `solve(grid)` function correctly."
        })
        
    return None

def solve_with_mcts(agent, sandbox, task_data, dsl_source, max_iterations=50):
    """Strategy 3: Tree Search with HDC Pruning"""
    print(f"   🌳 Strategy: MCTS (Neuro-Vector-Symbolic)...")
    
    # Initialize HDC Space (One-time cost)
    hdc = HDCSpace()
    solver = MCTSSolver(agent, sandbox, dsl_source, hdc_space=hdc)
    
    # Run Solver
    code = solver.solve(task_data, max_iterations=max_iterations)
    
    if code:
        print(f"   ✨ MCTS found a program!")
        return code
    else:
        print(f"   ❌ MCTS failed to find a path.")
        return None

def main():
    parser = argparse.ArgumentParser(description="ARC Solver using Qwen + LoRA")
    parser.add_argument("--mode", choices=["sampling", "refinement", "mcts"], default="refinement", help="Solving strategy")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples for sampling mode")
    parser.add_argument("--steps", type=int, default=3, help="Max steps for refinement mode")
    parser.add_argument("--task_file", type=str, default=None, help="Run on a specific task file (json)")
    parser.add_argument("--random", type=int, default=0, help="Run on N random tasks from training set")
    parser.add_argument("--save_dream", action="store_true", help="Save successful solutions to dream data")
    parser.add_argument("--mcts_iterations", type=int, default=50, help="Max iterations for MCTS")
    args = parser.parse_args()

    # Init Agent & Sandbox
    try:
        agent = UnslothAgent()
    except Exception as e:
        print(f"Critical Error: Failed to load model. {e}")
        return

    sandbox = LocalSandbox(timeout=5)
    dsl_source = inspect.getsource(dsl)
    
    # Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine tasks to solve
    tasks_to_solve = []
    if args.task_file:
         if os.path.exists(args.task_file):
             tasks_to_solve.append((os.path.basename(args.task_file), args.task_file))
         else:
             print(f"Error: File not found {args.task_file}")
             return
    else:
        arc_train_dir = config.ARC_TRAIN_DIR
        
        if args.random > 0:
            # Pick N random tasks
            all_tasks = list(arc_train_dir.glob("*.json"))
            if not all_tasks:
                 print(f"Error: No tasks found in {arc_train_dir}")
                 return
            selected = random.sample(all_tasks, min(args.random, len(all_tasks)))
            for path in selected:
                 tasks_to_solve.append((path.stem, str(path)))
        else:
            # Use predefined target tasks
            for task_id in TARGET_TASKS:
                 path = arc_train_dir / (task_id + ".json")
                 tasks_to_solve.append((task_id, str(path)))
    
    score = 0
    total = len(tasks_to_solve)
    
    for task_id, task_path in tasks_to_solve:
        print(f"\n🧩 [Task: {task_id}] Mode: {args.mode}")
        
        if not os.path.exists(task_path):
            print(f"⚠️ Task file not found: {task_path}")
            continue
            
        task_data = load_arc_task(task_path)
        
        if args.mode == "sampling":
            final_code = solve_with_sampling(agent, sandbox, task_data, dsl_source, num_samples=args.samples)
        elif args.mode == "mcts":
            final_code = solve_with_mcts(agent, sandbox, task_data, dsl_source, max_iterations=args.mcts_iterations)
        else:
            final_code = solve_with_refinement(agent, sandbox, task_data, dsl_source, max_steps=args.steps)
            
        if final_code:
            # Check Test Set
            test_inp = str(task_data['test'][0]['input'])
            expected = task_data['test'][0]['output']
            success, output, _ = sandbox.execute(final_code, test_input=test_inp)
            
            if success:
                try:
                    pred = ast.literal_eval(output.strip())
                    if pred == expected:
                        print(f"   🎉🎉🎉 Task {task_id} SOLVED (Test Set Passed) !!!")
                        score += 1
                        if args.save_dream:
                            save_successful_trace(task_id, final_code, task_data)
                        continue
                except: pass
            print(f"   ⚠️ Passed Training but Failed Test Set.")
        else:
            print("   💀 Failed to find a solution.")

    print(f"\n🏆 Final Score: {score}/{total}")

if __name__ == "__main__":
    main()

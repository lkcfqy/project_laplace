# src/mcts.py
"""
MCTS Solver Module
-----------------
This module implements a Monte Carlo Tree Search (MCTS) solver for ARC tasks.
It uses HDC for state similarity evaluation and an agent for node expansion.
"""
import random
import copy
from hdc import HDCSpace, GridEncoder
import dsl
import math
import torch
try:
    from value_net import HDCValueNet
    HAS_VALUE_NET = True
except ImportError:
    HAS_VALUE_NET = False
import json
import config
import os

class MCTSNode:
    def __init__(self, trace, grid, parent=None):
        self.trace = trace  # List of code lines executed so far
        self.grid = grid    # Current grid state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        
    def uct(self, exploration_weight=1.414):
        if self.visits == 0: return float('inf')
        return self.value / self.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSSolver:
    def __init__(self, agent, sandbox, dsl_source, hdc_space=None):
        self.agent = agent
        self.sandbox = sandbox
        self.dsl_source = dsl_source
        self.hdc = hdc_space if hdc_space else HDCSpace()
        self.dsl_source = dsl_source
        self.hdc = hdc_space if hdc_space else HDCSpace()
        self.encoder = GridEncoder(self.hdc)
        
        self.value_net = None
        if HAS_VALUE_NET:
             # Try to load weights if exist
             try:
                 self.value_net = HDCValueNet()
                 if os.path.exists(config.VALUE_NET_PATH):
                     self.value_net.load_state_dict(torch.load(config.VALUE_NET_PATH))
                     self.value_net.to(self.hdc.device)
                     self.value_net.eval()
                     print("   🧠 Neural Value Network loaded!")
                 else:
                     print("   ⚠️ Value Net initialized but no weights found. Using random init (not recommended) or please train first.")
                     self.value_net = None # Fallback until trained
             except Exception as e: 
                 print(f"   ⚠️ Failed to load Value Net: {e}")
                 self.value_net = None
        
    def solve(self, task_data, max_iterations=50):
        self.task_data = task_data # Store for use in expansion
        # Initial State: Input Grid of first example (Simplification for MCTS prototype)
        # Ideally we track state across all examples, but let's solve for Ex 1 first
        train_pairs = task_data['train']
        start_grid = train_pairs[0]['input']
        target_grid = train_pairs[0]['output']
        
        target_vec = self.encoder.encode_grid(target_grid)
        root = MCTSNode(trace=[], grid=start_grid)
        
        best_program = None
        best_score = -1.0
        
        print(f"   🌳 Starting MCTS (Target Sim Check activated)...")
        
        for i in range(max_iterations):
            node = root
            
            # 1. Selection
            while node.children and not node.is_terminal:
                node = max(node.children, key=lambda n: n.uct())
                
            # 2. Expansion
            if not node.is_terminal:
                new_nodes = self.expand_node(node, task_data)
                if new_nodes:
                    node = random.choice(new_nodes)
                else:
                    node.is_terminal = True # No valid moves
            
            # 3. Simulation (Rollout) & Evaluation
            # Evaluate using HDC similarity to target
            current_vec = self.encoder.encode_grid(node.grid)
            
            if self.value_net:
                # Use Neural Value
                with torch.no_grad():
                     # Assuming value_net takes text/vector. need to ensure dimensions match
                     sim = self.value_net(current_vec, target_vec).item()
            else:
                # Use Static Cosine Similarity
                sim = self.hdc.similarity(current_vec, target_vec)
            
            # Check if solved (perfect match in pixels for Example 1)
            if node.grid == target_grid:
                print(f"   ✨ Candidate found at iter {i}! Validating on ALL training examples...")
                
                # --- VALIDATION STEP ---
                candidate_code = self.construct_program(node.trace)
                full_program = f"{self.dsl_source}\n\n{candidate_code}"
                
                all_passed = True
                for idx, pair in enumerate(task_data['train']):
                    inp_grid = str(pair['input'])
                    exp_out = pair['output']
                    
                    # Run sandbox
                    success, output, error = self.sandbox.execute(full_program, test_input=inp_grid)
                    
                    if not success:
                        print(f"      ❌ Failed on Example {idx+1} (Runtime Error): {error}")
                        all_passed = False
                        break
                        
                    try:
                        pred_grid = eval(output.strip())
                        if pred_grid != exp_out:
                            print(f"      ❌ Failed on Example {idx+1} (Logic Error: Output mismatch)")
                            all_passed = False
                            break
                    except:
                        print(f"      ❌ Failed on Example {idx+1} (Parsing Error)")
                        all_passed = False
                        break
                
                if all_passed:
                    print(f"   🎉 Validated on {len(task_data['train'])} examples! Solution Accepted.")
                    
                    # --- Save HDC Training Data ---
                    self.save_hdc_trace(node, target_grid)
                    
                    return candidate_code
                else:
                    print(f"   ⚠️ Candidate rejected. Continue searching...")
                    # Penalize this node to avoid re-visiting immediately? 
                    # For now, just setting value low or treat as terminal failure for this path
                    node.value = -1.0 
                    # node.is_terminal = True # Optional: kill this branch
                
            # Pruning threshold
            reward = sim
            if sim < 0.5 and len(node.trace) > 1:
                reward = 0 # Prune bad branches heavily
                
            # 4. Backpropagation
            temp = node
            while temp:
                temp.visits += 1
                temp.value += reward
                temp = temp.parent
                
        return None

    def expand_node(self, node, task_data):
        # Call LLM to suggest next line of code
        current_code = "\n".join(node.trace)
        
        target_grid = node.grid
        rows = len(target_grid)
        cols = len(target_grid[0])
        
        SYMBOLS = {0: '.', 1: 'B', 2: 'R', 3: 'G', 4: 'Y', 5: 'X', 6: 'M', 7: 'O', 8: 'A', 9: 'C'}
        grid_str_lines = []
        for r in range(rows):
            line_str = "".join([SYMBOLS.get(c, str(c)) for c in target_grid[r]])
            grid_str_lines.append(f"{r:2d}| {line_str}")
        grid_ascii = "\n".join(grid_str_lines)

        diff_desc = "Grid changed from previous step." if node.parent else "Start State"
        if node.parent and node.grid == node.parent.grid:
            diff_desc = "⚠️ No change detected!"
        
        examples_str = ""
        if 'train' in task_data:
            for i, pair in enumerate(task_data['train'][:3]):
                examples_str += f"\nExample {i+1} Input:\n{pair['input']}\nExample {i+1} Output:\n{pair['output']}\n"

        # Analyze color diff
        out_colors = set()
        for p in task_data['train']:
            for row in p['output']:
                for c in row: out_colors.add(c)
        target_colors = sorted(list(out_colors))

        prompt = f"""[ARC MISSION]
Goal: Solve the task based on these examples:
{examples_str}

[CURRENT STATE]
Grid:
{grid_ascii}
(Target colors: {target_colors})
Diff: {diff_desc}

Current Trace:
{current_code if current_code else "pass"}

[INSTRUCTION]
Suggest 3 DIFFERENT Code Blocks to reach the goal.
Use '---' to separate ideas.

IMPORTANT: The code you write must be a GENERAL rule that works for ALL examples above, not just the Current State grid.
For instance, if you change color 2 to 3, ensure that logic applies universally.

CRITICAL RULES:
1. Every code block MUST modify the variable `grid`. Counting or analysis is NOT allowed as a standalone step.
2. Maintain valid Python indentation for loops and IF statements.
3. You can use variables (like `objs`) as long as the final result is assigned to `grid`.

[FORMAT]
Thought: (Analysis)
Code:
grid = ...
---
"""
        messages = [
            {"role": "system", "content": "You are a Python ARC expert. Output logic blocks followed by '---'."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.agent.generate_code(messages, max_new_tokens=400, temperature=0.7)
        print(f"\n⚡ [CLINIC DIAGNOSIS] ⚡\nLLM Output:\n{response}\n-------------------------")
        
        candidates = [c.strip() for c in response.split('---') if c.strip()]
        children = []
        for block in candidates[:3]:
            if "Code:" in block: block = block.split("Code:", 1)[1]
            
            # --- Robust Indentation-Aware Extractor ---
            raw_lines = block.split('\n')
            code_lines = []
            
            # First pass: find first line that looks like real code (not a label)
            # and detect its base indentation
            base_indent = -1
            for line in raw_lines:
                stripped = line.strip().replace("`", "")
                if not stripped: continue
                if any(stripped.lower().startswith(p) for p in ["thought", "idea", "code:", "suggestion"]): continue
                
                # First code line found
                if base_indent == -1:
                    base_indent = len(line) - len(line.lstrip())
                
                # Normalize indentation relative to base_indent
                rel_line = line[base_indent:] if len(line) >= base_indent else line.lstrip()
                code_lines.append(rel_line.replace("`", ""))
            
            if not code_lines: continue
            
            # --- Mutation Filter ---
            # Rejoin to check if 'grid' is being assigned to
            full_block_str = "\n".join(code_lines)
            if "grid =" not in full_block_str and "grid[" not in full_block_str:
                 # Attempt auto-fix for single liners
                 if len(code_lines) == 1 and "(" in code_lines[0] and "=" not in code_lines[0]:
                     code_lines[0] = f"grid = {code_lines[0]}"
                 else:
                     continue # Reject no-op code

            clean_block = "\n".join(code_lines) # Keep it raw, let construct_program handle indent
            
            # Construct candidate program using helper to ensure consistent indentation
            candidate_trace = node.trace + [clean_block]
            full_program_code = self.construct_program(candidate_trace)
            full_program = f"{self.dsl_source}\n\n{full_program_code}"
            
            # --- FIX: Execute on ORIGINAL Input (Example 1) ---
            # Previously: executed on node.grid (which is already intermediate state)
            # causing double execution of previous steps.
            start_grid = self.task_data['train'][0]['input']
            
            success, _, _ = self.sandbox.execute(full_program, test_input=str(start_grid))
            if success:
                try:
                    wrapped_run = full_program + f"\nprint(solve({start_grid}))"
                    s, o, e = self.sandbox.execute(wrapped_run, test_input="")
                    if s and o.strip():
                        new_grid = eval(o.strip())
                        if isinstance(new_grid, list):
                            child = MCTSNode(trace=node.trace + [clean_block], grid=new_grid, parent=node)
                            node.children.append(child)
                            children.append(child)
                except: pass
        return children

    def construct_program(self, trace):
        # Indent each line of the trace
        indented_lines = []
        for block in trace:
            for line in block.split('\n'):
                indented_lines.append(f"    {line}")
        
        if not indented_lines:
            indented_lines.append("    pass")
            
        indented_lines.append("    return grid")
            
        return "def solve(grid):\n" + "\n".join(indented_lines)

    def save_hdc_trace(self, successful_node, target_grid):
        """Backtrack from winning node and save (State, Target, Label=1)."""
        temp = successful_node
        entries = []
        
        while temp:
             # Positive Sample: This state led to the solution
             entry = {
                 "state_grid": temp.grid,
                 "target_grid": target_grid,
                 "label": 1.0
             }
             entries.append(entry)
             
             # Negative Samples: Siblings that were NOT chosen (or visited but didn't win)
             # Note: This is a simplification. Siblings might also be valid paths but we didn't pick them.
             # Ideally we check their values. For now, we only trust the path we took.
             # We can add siblings with label 0.0 if we assume unique solution path (risky in ARC).
             # Let's stick to Positive samples for now to train 'similarity'.
             # Or better: random other grids as negatives? 
             # Let's start with just Positives to learn "What looks close to target".
             # Actually, to train a discriminator/similarity, we NEED negatives.
             # Let's use siblings as negatives.
             if temp.parent:
                 for sibling in temp.parent.children:
                     if sibling != temp:
                         entries.append({
                             "state_grid": sibling.grid,
                             "target_grid": target_grid,
                             "label": 0.0 # Heuristic: Didn't lead to solution *in this run*
                         })
             
             temp = temp.parent
             
        # Append to file
        with open(config.HDC_DATA_FILE, "a") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"   💾 Saved {len(entries)} HDC training samples.")

import subprocess
import time
import os
import config

def run_wake_phase(iteration):
    print(f"\n☀️  [Cycle {iteration}] WAKE Phase: Solving Tasks...")
    # Run solver on a few target tasks
    # In a real scenario, this would iterate over many tasks.
    # Here we run on a RANDOM set of tasks to maximize chance of finding easier ones
    # to bootstrap the learning process.
    cmd = ["python", "src/solve.py", "--mode", "mcts", "--save_dream", "--random", "5"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Wake phase complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Wake phase failed with error: {e}")

def run_sleep_phase(iteration):
    print(f"\n🌙 [Cycle {iteration}] SLEEP Phase: Learning from Dreams...")
    
    has_dream = False
    if os.path.exists(config.DREAM_DATA_FILE):
        with open(config.DREAM_DATA_FILE, 'r') as f:
            if f.readlines():
                has_dream = True
    
    # Check for synthetic data
    has_synthetic = os.path.exists(config.DSL_DATA_FILE)
    
    if not has_dream and not has_synthetic:
        print("⚠️ No data found (neither dreams nor synthetic). Skipping sleep.")
        return

    if has_dream:
        print(f"   🧠 Found Dream Data.")
    if has_synthetic:
        print(f"   🧬 Found Synthetic Data.")

    # Run training
    # For sleep phase, we might want fewer steps or specific Learning Rate
    # But train.py defaults are fine for now (500 steps).
    cmd = ["python", "src/train.py"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Sleep phase (Language Model) complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Sleep phase (LLM) failed: {e}")

    # --- ENHANCEMENT: Train Value Net (Puzzle 4) ---
    print(f"   🧠 Training HDC Value Network...")
    cmd_hdc = ["python", "src/train_value_net.py"]
    try:
        subprocess.run(cmd_hdc, check=True)
        print("✅ Sleep phase (Value Net) complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Sleep phase (Value Net) failed: {e}")

def bootstrap(cycles=1):
    print("🚀 Starting Bootstrap Wake-Sleep Loop...")
    for i in range(1, cycles + 1):
        run_wake_phase(i)
        run_sleep_phase(i)
        print(f"\n🔁 Cycle {i} finished.")

if __name__ == "__main__":
    # Continuous learning loop
    bootstrap(cycles=100)

# 🌌 Project Laplace ✨

Welcome to **Project Laplace**, an autonomous agent system based on **Neuro-Symbolic AI** designed to solve the **ARC (Abstraction and Reasoning Corpus)** tasks through Program Synthesis! 🧩✨

This project beautifully combines **Large Language Models (LLMs)**, **Monte Carlo Tree Search (MCTS)**, and **Hyperdimensional Computing (HDC)**, achieving self-evolution through a fascinating **"Wake-Sleep"** cycle mechanism. 🌗

## ✨ Core Features 🛠️

* **🧠 Neuro-Symbolic Architecture**: Uses a fine-tuned **Qwen2.5-Coder** to generate Python code, combined with a predefined **DSL (Domain Specific Language)** to manipulate grids, ensuring logical interpretability and precision! 🎯
* **🔁 Wake-Sleep Guided Learning**:
  * **☀️ Wake Phase**: Employs MCTS and the current model to attempt solving ARC tasks, generating successful trajectories known as "Dreams". ☁️
  * **🌙 Sleep Phase**: Utilizes the "dream" data (successful problem-solving trajectories) and synthetic data to perform LoRA fine-tuning on the LLM and train the value network. 💤
* **🌳 MCTS + HDC**: Implements a Monte Carlo Tree Search solver, utilizing **Hyperdimensional Computing (HDC)** for efficient state similarity evaluation and Neural Value Network guided search pruning. ✂️
* **🛡️ Safe Sandbox Environment**: Features a built-in **Docker Sandbox** and AST static checks to ensure generated code runs safely in an isolated environment. 🔒
* **🚀 Unsloth Acceleration**: Leverages the Unsloth framework for efficient 4-bit quantized LoRA fine-tuning, drastically reducing VRAM requirements (perfect for consumer GPUs like the RTX 30 series!). 💻

## 📂 Project Structure 📁

```text
project_laplace/
├── data/                   # Stores training data, the ARC dataset, and generated "Dreams"
├── models/                 # Stores LoRA adapters and Value Net weights
├── src/                    # Source code directory
│   ├── agent_lora.py       # LLM Agent wrapper (Unsloth/Qwen)
│   ├── bootstrap_loop.py   # Main entry for the Wake-Sleep loop
│   ├── config.py           # Project path configurations
│   ├── dataset_hdc.py      # Data loader for HDC training
│   ├── docker_sandbox.py   # Docker sandbox for safe code execution
│   ├── dsl.py              # Domain Specific Language (DSL) primitives for ARC tasks
│   ├── executor.py         # Code executor (includes AST safety checks)
│   ├── gen_baseline.py     # Generator for synthetic baseline data (Reverse engineering DSL)
│   ├── gen_synthetic_tasks.py # Generates specific types of synthetic ARC tasks
│   ├── hdc.py              # Hyperdimensional Computing (HDC) implementation
│   ├── mcts.py             # Monte Carlo Tree Search solver
│   ├── solve.py            # Standalone solving script (supports Sampling/Refinement/MCTS)
│   ├── train.py            # LLM LoRA fine-tuning script
│   ├── train_value_net.py  # Script to train the HDC Value Network
│   └── value_net.py        # Value Network model definition
└── .gitignore

```

## 🛠️ Installation & Environment 💻

### Prerequisites 📌

* Python 3.10+ 🐍
* NVIDIA GPU (CUDA supported, 8GB+ VRAM recommended) 🎮
* Docker (for the safe sandbox; optional but highly recommended!) 🐳

### Installing Dependencies 📦

1. **Clone the project** 📥

```bash
git clone [https://github.com/your-username/project_laplace.git](https://github.com/your-username/project_laplace.git)
cd project_laplace

```

2. **Install Python dependencies** 🪄
*(Using Conda or venv is recommended)*

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install unsloth "unsloth[colab-new]"  # Or follow the official Unsloth docs
pip install docker transformers datasets trl peft bitsandbytes

```

3. **Prepare the Data** 📊

* Ensure that the ARC dataset (`arc/training` and `arc/evaluation`) is inside the `data/` directory.
* Alternatively, run the generation scripts first to create synthetic data.

## 🚀 Quick Start 🏃‍♀️💨

### 1. Generate Synthetic Data (Cold Start) 🧊

When starting with no model weights, generate some synthetic data first to teach the model the basic DSL usage!

```bash
# Generate basic DSL training samples
python src/gen_baseline.py

# Generate simple synthetic ARC tasks to test the solver
python src/gen_synthetic_tasks.py --count 50

```

### 2. Start the Wake-Sleep Loop (Recommended) 🌞🌛

This is the main operation mode of the project. It automatically alternates between "Task Solving" and "Model Training".

```bash
# Loops 100 times by default
python src/bootstrap_loop.py

```

### 3. Run the Solver Standalone 🕵️‍♀️

If you just want to test the model's ability to solve specific tasks:

```bash
# Solve a specific task using MCTS mode
python src/solve.py --mode mcts --task_file data/arc/training/25d8a9c8.json

# Use Refinement mode (Self-correction)
python src/solve.py --mode refinement --steps 3 --random 5

```

### 4. Standalone Training 🏋️‍♂️

To manually trigger the training process:

```bash
# Train the LLM (Qwen LoRA)
python src/train.py

# Train the HDC Value Network
python src/train_value_net.py

```

## 🧠 Core Logic Explained 💡

### DSL (Domain Specific Language) 🔠

Defined in `src/dsl.py`, it contains high-level abstract functions commonly used for ARC tasks, such as:

* `get_objects(grid)`: Object detection 🔍
* `flood_fill(grid, r, c, color)`: Flood fill 🌊
* `move_object(grid, obj, dr, dc)`: Move object 📦
* `detect_periodicity(grid)`: Periodicity detection 🔁

### MCTS Solver 🌳

Defined in `src/mcts.py`.

1. **Selection**: Uses the UCT algorithm to select the most promising nodes. ✨
2. **Expansion**: Calls the LLM (`UnslothAgent`) to generate Python code suggestions based on the current grid state. 📝
3. **Evaluation**:
* Executes code within the sandbox. 🛡️
* Uses **HDC (src/hdc.py)** to encode the grid state. 🧩
* Calculates cosine similarity between the current and target states, or predicts the success rate via the **Value Net**. ⚖️


4. **Backprop**: Updates the path values based on the evaluation. 🔙

## ⚠️ Important Notes 🚨

* **VRAM Usage**: The default configuration is optimized for 24GB VRAM. If your VRAM is smaller (e.g., 8GB-12GB), please lower the `batch_size` in `src/train.py` and ensure `load_in_4bit=True` is enabled! 📉
* **Docker Permissions**: Running the sandbox requires the current user to have Docker permissions (on Linux, this usually means adding the user to the `docker` group). If Docker isn't used, the program will fall back to local execution, which carries security risks! ⚠️

## 🤝 Contribution 💖

Issues and Pull Requests are incredibly welcome! Help us improve the DSL library, optimize MCTS strategies, or enhance the Value Net. 🙌

## 📜 License 📄

MIT License

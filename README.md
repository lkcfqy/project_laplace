# 🌌 Project Laplace (拉普拉斯计划)

**Project Laplace** 是一个基于 **神经符号 AI (Neuro-Symbolic AI)** 的自主智能体系统，旨在通过程序合成（Program Synthesis）解决 **ARC (Abstraction and Reasoning Corpus)** 抽象推理任务。

该项目结合了 **大语言模型 (LLM)**、**蒙特卡洛树搜索 (MCTS)** 和 **超维计算 (Hyperdimensional Computing, HDC)**，并通过 **"Wake-Sleep"（清醒-睡眠）** 循环机制实现自我进化。

## ✨ 核心特性

* **🧠 神经符号架构 (Neuro-Symbolic)**: 使用微调后的 **Qwen2.5-Coder** 生成 Python 代码，结合预定义的 **DSL (领域特定语言)** 来操作网格，保证逻辑的可解释性和精确性。
* **🔁 Wake-Sleep 引导式学习**:
* **☀️ Wake Phase (清醒阶段)**: 使用 MCTS 和当前模型尝试解决 ARC 任务，生成成功的轨迹（Dreams）。
* **🌙 Sleep Phase (睡眠阶段)**: 利用“梦境”数据（成功的解题轨迹）和合成数据对 LLM 进行 LoRA 微调，并训练价值网络。


* **🌳 MCTS + HDC**: 实现了蒙特卡洛树搜索求解器，利用 **超维计算 (HDC)** 进行高效的状态相似度评估和神经价值网络（Value Net）引导搜索剪枝。
* **🛡️ 安全沙箱环境**: 内置 **Docker Sandbox** 和 AST 静态检查，确保生成的代码在隔离环境中安全执行。
* **🚀 Unsloth 加速**: 利用 Unsloth 框架进行高效的 4-bit 量化 LoRA 微调，大幅降低显存需求（适配 RTX 30系列等消费级显卡）。

## 📂 项目结构

```text
project_laplace/
├── data/                   # 存放训练数据、ARC 数据集和生成的"梦境"
├── models/                 # 存放 LoRA 适配器和 Value Net 权重
├── src/                    # 源代码目录
│   ├── agent_lora.py       # LLM 代理封装 (Unsloth/Qwen)
│   ├── bootstrap_loop.py   # Wake-Sleep 主循环入口
│   ├── config.py           # 项目路径配置
│   ├── dataset_hdc.py      # HDC 训练数据加载器
│   ├── docker_sandbox.py   # Docker 代码执行沙箱
│   ├── dsl.py              # ARC 领域特定语言原语 (DSL)
│   ├── executor.py         # 代码执行器 (含 AST 安全检查)
│   ├── gen_baseline.py     # 合成基准数据生成器 (逆向工程 DSL)
│   ├── gen_synthetic_tasks.py # 生成特定类型的合成 ARC 任务
│   ├── hdc.py              # 超维计算 (HDC) 实现
│   ├── mcts.py             # 蒙特卡洛树搜索求解器
│   ├── solve.py            # 单次求解脚本 (支持 Sampling/Refinement/MCTS)
│   ├── train.py            # LLM LoRA 微调脚本
│   ├── train_value_net.py  # HDC 价值网络训练脚本
│   └── value_net.py        # 价值网络模型定义
└── .gitignore

```

## 🛠️ 安装与环境

### 前置要求

* Python 3.10+
* NVIDIA GPU (支持 CUDA，建议 8GB+ 显存)
* Docker (用于安全沙箱，可选但推荐)

### 安装依赖

1. **克隆项目**
```bash
git clone https://github.com/your-username/project_laplace.git
cd project_laplace

```


2. **安装 Python 依赖**
*建议使用 Conda 或 venv*
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth "unsloth[colab-new]"  # 或根据 unsloth 官方文档安装
pip install docker transformers datasets trl peft bitsandbytes

```


3. **准备数据**
* 确保 `data/` 目录下有 ARC 数据集（`arc/training` 和 `arc/evaluation`）。
* 或者先运行生成脚本生成合成数据。



## 🚀 快速开始

### 1. 生成合成数据 (Cold Start)

在没有任何模型权重时，首先生成一些合成数据来教会模型基本的 DSL 用法。

```bash
# 生成基础 DSL 训练样本
python src/gen_baseline.py

# 生成简单的合成 ARC 任务用于测试求解器
python src/gen_synthetic_tasks.py --count 50

```

### 2. 启动 Wake-Sleep 循环 (推荐)

这是项目的主要运行模式。它会自动交替进行“求解任务”和“模型训练”。

```bash
# 默认循环 100 次
python src/bootstrap_loop.py

```

### 3. 单独运行求解器

如果你只想测试模型解决特定任务的能力：

```bash
# 使用 MCTS 模式求解特定任务
python src/solve.py --mode mcts --task_file data/arc/training/25d8a9c8.json

# 使用 Refinement (自我修正) 模式
python src/solve.py --mode refinement --steps 3 --random 5

```

### 4. 单独训练

手动触发训练流程：

```bash
# 训练 LLM (Qwen LoRA)
python src/train.py

# 训练 HDC 价值网络
python src/train_value_net.py

```

## 🧠 核心逻辑详解

### DSL (领域特定语言)

定义在 `src/dsl.py` 中，包含 ARC 任务常用的高层抽象函数，如：

* `get_objects(grid)`: 对象检测
* `flood_fill(grid, r, c, color)`: 泛洪填充
* `move_object(grid, obj, dr, dc)`: 移动对象
* `detect_periodicity(grid)`: 周期性检测

### MCTS Solver

定义在 `src/mcts.py`。

1. **选择 (Selection)**: 使用 UCT 算法选择最有潜力的节点。
2. **扩展 (Expansion)**: 调用 LLM (`UnslothAgent`) 根据当前网格状态生成 Python 代码建议。
3. **模拟与评估 (Evaluation)**:
* 在沙箱中执行代码。
* 使用 **HDC (src/hdc.py)** 编码网格状态。
* 计算当前状态与目标状态的余弦相似度，或通过 **Value Net** 预测成功率。


4. **反向传播 (Backprop)**: 更新路径价值。

## ⚠️ 注意事项

* **显存占用**: 默认配置针对 24GB 显存优化，如果显存较小 (如 8GB-12GB)，请在 `src/train.py` 中将 `batch_size` 调小，并确保 `load_in_4bit=True`。
* **Docker 权限**: 运行沙箱需要当前用户有 Docker 权限 (Linux 下通常需要加入 `docker` 用户组)。如果不使用 Docker，程序会回退到本地执行，但存在安全风险。

## 🤝 贡献

欢迎提交 Issue 和 Pull Requests 来改进 DSL 库、优化 MCTS 策略或增强 Value Net。

## 📜 许可证

MIT License

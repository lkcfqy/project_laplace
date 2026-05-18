# project_laplace

Project Laplace 是一个面向 ARC 任务的神经符号程序合成实验仓库。它尝试把 Qwen2.5-Coder/LoRA、ARC DSL、采样/自修复/MCTS 搜索、HDC 状态评估、安全执行和 wake-sleep 训练循环组合起来，用代码生成的方式求解抽象推理任务。

## 当前状态

仓库包含 DSL、求解器、MCTS、HDC 编码、Unsloth LoRA 训练、value net 训练和 bootstrap loop。默认适合在有 CUDA 的本地机器上做实验；如果没有 `models/qwen_dsl_adapter`，`UnslothAgent` 会尝试加载 `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` 作为冷启动模型。

ARC 数据集和训练产物默认不随仓库提交，需要放到 `data/arc/training`、`data/arc/evaluation`、`models/` 等路径下。

## 主要模块

- `src/dsl.py`：ARC 网格操作 DSL，包括物体检测、裁剪、填充、旋转、翻转、移动、合并、周期检测等原语。
- `src/solve.py`：求解入口，支持 `sampling`、`refinement` 和 `mcts` 三种模式。
- `src/mcts.py`：MCTS 搜索，结合 HDC 相似度或 value net 做状态评估。
- `src/hdc.py`：网格到高维向量的编码。
- `src/executor.py` / `src/docker_sandbox.py`：代码安全检查和沙箱执行。
- `src/agent_lora.py`：Unsloth 模型加载和代码生成封装。
- `src/train.py`：LoRA 微调，支持 synthetic data 和 dream traces。
- `src/bootstrap_loop.py`：wake-sleep 循环，先求解再用成功轨迹继续训练。

## 环境准备

建议使用 Python 3.10+、CUDA 和较大的显存环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install transformers datasets trl peft bitsandbytes unsloth docker
```

如果启用 Docker 沙箱，请确认 Docker daemon 可用，且当前用户有权限启动容器。

## 运行示例

指定 ARC 任务并用 MCTS 求解：

```bash
python src/solve.py --mode mcts --task_file data/arc/training/25d8a9c8.json
```

使用 refinement 模式：

```bash
python src/solve.py --mode refinement --steps 3 --task_file data/arc/training/25d8a9c8.json
```

启动 wake-sleep 循环：

```bash
python src/bootstrap_loop.py
```

## 生成与训练数据

配置路径集中在 `src/config.py`：

- `data/synthetic_dsl_train.jsonl`
- `data/synthetic_primitives_train.jsonl`
- `data/dream_traces.jsonl`
- `data/hdc_training_data.jsonl`
- `models/qwen_dsl_adapter`
- `models/value_net.pth`

成功解出的任务可通过 `--save_dream` 写入 dream traces，后续由 `src/train.py` 用于增量训练。

## 注意事项

- `src/dsl.py` 中仍有少量原型代码痕迹，例如重复 `return counts`，不影响主要阅读但说明项目仍在快速实验阶段。
- MCTS 当前以第一个训练样例作为初始搜索状态，再对所有训练样例验证；这是一种原型化近似。
- 生成代码会被 AST 检查和沙箱执行过滤，但仍不应在不受控环境中运行未知生成代码。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。

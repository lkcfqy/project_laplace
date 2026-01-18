# src/train.py
"""
Training Module
--------------
This module handles the fine-tuning of the model using LoRA.
It loads the dataset, configures the model and tokenizer, and runs the training loop.
"""
import os
import torch
import torch._inductor.config
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import config

# --- 配置 ---
# 路径锁死逻辑 (使用 config 替代)
# output_dir moved to inside or used from config
output_dir = str(config.ADAPTER_PATH)

# 模型配置
max_seq_length = 2048 # 允许的上下文长度
dtype = None # 自动检测 (Float16 for 3080)
load_in_4bit = True # 4bit 量化 (关键！省显存)

def train():
    print(f"🚀 开始训练流程...")
    
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_exists = os.path.exists(config.ADAPTER_PATH)
    if adapter_exists:
        print(f"🔄 Loading existing adapter from {config.ADAPTER_PATH} for incremental training...")
        model_name = str(config.ADAPTER_PATH)

    # 1. 加载模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. 添加 LoRA 适配器 (给模型加外挂)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # LoRA 秩 (越大越聪明，但显存占用越高，16 是 3080 的甜点)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth", # 显存优化技术
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. 准备数据格式
    # Load both datasets
    print("📥 Loading datasets...")
    print("📥 Loading datasets...")
    dsl_data_file = str(config.DSL_DATA_FILE)
    primitives_data_file = str(config.PRIMITIVES_DATA_FILE)
    dream_data_file = str(config.DREAM_DATA_FILE)
    
    dataset_files = []
    
    # 策略：如果 Adapter 已存在，说明是"复习"阶段，不再加载巨大的合成数据
    adapter_exists = os.path.exists(config.ADAPTER_PATH)
    
    if not adapter_exists:
        print("👶 Cold Start: Loading Synthetic Data for initial training...")
        if os.path.exists(dsl_data_file):
            dataset_files.append(dsl_data_file)
        if os.path.exists(primitives_data_file):
            dataset_files.append(primitives_data_file)
    else:
        print("🎓 Adapter found. Skipping synthetic data to save time (Incremental Learning).")

    if os.path.exists(dream_data_file):
        print(f"😴 Found Dream Data! Including {dream_data_file} in training.")
        dataset_files.append(dream_data_file)
    else:
        # 如果既没有 Adapter 又没有 Dream，或者有 Adapter 但没 Dream
        if not dataset_files and adapter_exists:
            print("⚠️ No new dreams to learn and synthetic data skipped. Exiting.")
            return

    if not dataset_files:
         raise FileNotFoundError("No training data found in data/ directory!")

    dataset = load_dataset("json", data_files=dataset_files, split="train")
    
    # Shuffle the dataset to mix task types
    dataset = dataset.shuffle(seed=42)
    
    # 定义格式化函数：把 User/Assistant 变成模型能读的 Prompt
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            # 提取 System, User, Assistant
            system_text = convo[0]['content']
            user_text = convo[1]['content']
            assistant_text = convo[2]['content']
            
            # 构造 Qwen 的 Chat 模板格式
            text = f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 4. 设置训练参数
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # 可以设为 True 加速，但容易爆显存
        args = TrainingArguments(
            per_device_train_batch_size = 2, # 3080 显存小，设为 2
            gradient_accumulation_steps = 4, # 累积梯度，相当于 batch_size = 8
            warmup_steps = 5,
            max_steps = 100 if adapter_exists else 500, # 复习时步数只需要很少，初学时多一点
                            # 正式训练建议设为 300 - 500 步
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit", # 8bit 优化器，省显存
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    # 5. 开始训练
    print("🔥 正在点火 (Training Started)...")
    trainer_stats = trainer.train()

    # 6. 保存模型 (LoRA Adapter)
    print(f"💾 正在保存 LoRA 适配器到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 同时也保存为 GGUF 格式 (可选，方便 Ollama 加载，这里先跳过，后续教你转换)
    print("✅ 训练完成！")

if __name__ == "__main__":
    train()
# src/agent_lora.py
"""
LoRA Agent Module
----------------
This module defines the UnslothAgent class, which loads a requested LoRA adapter
and performs inference to generate code based on input messages.
"""
import os
import torch
try:
    import torch._inductor.config
except ImportError:
    pass
from unsloth import FastLanguageModel
import re
import config

class UnslothAgent:
    def __init__(self):
        # 1. 确定模型路径
        adapter_path = str(config.ADAPTER_PATH)
        
        print(f"🧠 正在加载微调后的模型: {adapter_path} ...")
        
        if not os.path.exists(adapter_path):
            print(f"⚠️ Adapter path not found: {adapter_path}")
            print("   ⚠️ Loading Base Model (Qwen2.5-Coder-7B-Instruct) for Cold Start...")
            model_name = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
        else:
            model_name = adapter_path

        try:
            # 2. 加载模型
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = 8192,
                dtype = None,
                load_in_4bit = True,
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e
        
        # 3. 开启推理模式 (加速 2 倍)
        FastLanguageModel.for_inference(self.model)
        print("✅ 模型加载完成！")

    def generate_code(self, messages, max_new_tokens=4096, temperature=0.7):
        """
        输入：标准的 Chat 消息列表 [{"role": "user", "content": "..."}, ...]
        输出：代码字符串
        """
        # 使用 tokenizer 应用聊天模板
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # 加上 <|im_start|>assistant
            return_tensors = "pt",
        ).to("cuda")

        # 生成
        outputs = self.model.generate(
            input_ids = inputs,
            max_new_tokens = max_new_tokens,
            use_cache = True,
            temperature = temperature, 
            top_p = 0.9,
            do_sample = True if temperature > 0 else False,
        )
        
        # 解码
        response = self.tokenizer.batch_decode(outputs)
        response_text = response[0]
        
        # 提取 Assistant 的回答部分
        parts = response_text.split("<|im_start|>assistant\n")
        if len(parts) > 1:
            content = parts[-1].replace("<|im_end|>", "").strip()
        else:
            content = response_text

        # 提取代码块
        code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 暴力清洗
        if "```" in content:
            return content.replace("```python", "").replace("```", "").strip()
            
        return content.strip()
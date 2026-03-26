from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 把项目根目录加入搜索路径

import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
from training.dataset_loader import format_prompt, InstructionSample

def load_model_and_tokenizer(base_model_path: str, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,   #避免模型无限写，会在合理位置停下
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除 prompt 部分，只返回回答
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text

def main():
    parser = argparse.ArgumentParser(description="Compare different LoRA experts on the same query.")
    parser.add_argument("--base_model_path", type=str, default="models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--domains", type=str, nargs="+", default=["coding", "paper", "speech"], help="Domains to compare")
    parser.add_argument("--query", type=str, required=True, help="User query to test")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"QUERY: {args.query}")
    print(f"{'='*60}\n")

    sample = InstructionSample(instruction=args.query, input="", output="", meta={})
    formatted_prompt = format_prompt(sample)

    # 1. 加载基础模型并获取输出
    print(f"Loading Base Model: {args.base_model_path}...")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_path)
    base_model.eval()

    print("\n[BASE MODEL RESPONSE]")
    base_res = generate_response(base_model, tokenizer, formatted_prompt, args.max_new_tokens)
    print(base_res)
    print("-" * 40)
    
    # 释放 Base Model 显存，为后续 LoRA 腾空间（可选，取决于显存大小，但为了纯净性建议重新加载）
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 循环加载每个 LoRA 并获取输出（每次都重新加载 Base Model）
    for domain in args.domains:
        ROOT = Path(__file__).resolve().parent
        PROJECT_ROOT = Path(__file__).parent.parent  # 从 inference/ 回到项目根目录
        adapter_path = PROJECT_ROOT / "models/lora" / f"{domain}_lora"

        if not Path(adapter_path).exists():
            print(f"\nSkipping {domain}: Adapter not found at {adapter_path}")
            continue
        
        print(f"\n[LORA EXPERT: {domain.upper()} RESPONSE]")
        try:
            # 重新加载纯净的 Base Model
            # print(f"Reloading Base Model for {domain}...")
            model, tokenizer = load_model_and_tokenizer(args.base_model_path)
            
            model.eval()  # 先让基础模型 eval
            
            # 加载 LoRA
            lora_model = PeftModel.from_pretrained(model, adapter_path)
            lora_model.eval()

            # 生成
            lora_res = generate_response(lora_model, tokenizer, formatted_prompt, args.max_new_tokens)
            print(lora_res)
            
            # 彻底清理
            del lora_model
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {domain}: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()

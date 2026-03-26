import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from router.router_embedding import EmbeddingRouter

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
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text

DOMAIN_PROMPTS = {
    "paper": (
        "You are a research assistant specializing in academic papers. "
        "Please answer in a formal academic style with structured points. "
        "Use numbered lists for contributions when appropriate. "
        "Begin directly with the answer.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
    "coding": (
        "You are an expert programmer. "
        "Provide clear, efficient, and well-documented code solutions. "
        "Include comments in the code and example usage when helpful.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
    "speech": (
        "You are a speech technology expert. "
        "Explain speech processing concepts clearly and accurately.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
}

def get_prompt_template(domain: str) -> str:
    return DOMAIN_PROMPTS.get(domain, "Instruction: {instruction}\nInput: \nAnswer:")


def main():
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline with Multi-LoRA routing.")
    parser.add_argument("--base_model_path", type=str, default="models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--query", type=str, required=True, help="User query to test")
    parser.add_argument("--domain", type=str, default=None, choices=["coding", "paper", "speech"], help="Force a specific domain expert, skipping the router.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # --- 1. Router 决策 (如果用户没有强制指定) ---
    if args.domain:
        best_domain = args.domain
        print(f"Step 1: Skipping router, expert '{best_domain}' is manually selected.")
    else:
        print("Step 1: Routing query...")
        router = EmbeddingRouter()
        best_domain, scores = router.route(args.query)
        print(f"Selected Domain: {best_domain}")
        print(f"Scores: {scores}")
    print("-" * 40)

    # --- 2. 加载模型与 LoRA ---
    print(f"Step 2: Loading Base Model and LoRA for domain '{best_domain}'...")
    adapter_path = f"models/lora/{best_domain}_lora"
    if not Path(adapter_path).exists():
        print(f"Error: Adapter for domain '{best_domain}' not found at {adapter_path}")
        print("Falling back to Base Model.")
        model, tokenizer = load_model_and_tokenizer(args.base_model_path)
    else:
        model, tokenizer = load_model_and_tokenizer(args.base_model_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Successfully loaded LoRA from {adapter_path}")
    print("-" * 40)

    # --- 3. 生成回答 ---
    print("Step 3: Generating response...")
    prompt_template = get_prompt_template(best_domain)
    formatted_prompt = prompt_template.format(instruction=args.query)

    response = generate_response(model, tokenizer, formatted_prompt, args.max_new_tokens)

    print("\n" + "="*60)
    print("Final Answer:")
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()

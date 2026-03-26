from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "models/TinyLlama-1.1B-Chat-v1.0"
lora = "models/lora/coding_lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora)

prompt = """
Instruction: Write a Python function to find the maximum number in a list.".

Input:

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating...")
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.2,
    do_sample=True
)

print("\n=== MODEL OUTPUT ===\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
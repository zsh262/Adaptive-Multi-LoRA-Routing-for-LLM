from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


from training.dataset_loader import build_sft_text, format_prompt, load_instruction_dataset


# 领域自适应 LoRA 配置
DOMAIN_LORA_CONFIG = {
    "coding": dict(r=16, alpha=32),
    "paper": dict(r=16, alpha=32),
    "speech": dict(r=12, alpha=24),
    "default": dict(r=8, alpha=16),
}


@dataclass(frozen=True)
class TrainConfig:
    domain: str
    base_model_path: str
    train_path: str
    eval_path: str | None
    output_dir: str
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    max_steps: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    seed: int
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    load_in_8bit: bool
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    device_map: str | None
    resume_from_checkpoint: str | None
    eval_only: bool
    adapter_path: str | None
    max_train_samples: int | None
    max_eval_samples: int | None


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=str, required=True, help="Training domain (e.g., coding, paper, speech)")
    p.add_argument("--base_model_path", type=str, default="models/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--eval_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # 修复 action="store_true" 的用法
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    p.set_defaults(fp16=True, gradient_checkpointing=True, load_in_4bit=True)

    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--adapter_path", type=str, default=None)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    a = p.parse_args()
    
    domain = a.domain
    train_path = a.train_path or f"data/{domain}/train.json"
    eval_path = a.eval_path or f"data/{domain}/val.json"
    output_dir = a.output_dir or f"models/lora/{domain}_lora"
    
    if eval_path and eval_path.lower() == "none":
        eval_path = None

    return TrainConfig(
        domain=domain,
        base_model_path=a.base_model_path,
        train_path=train_path,
        eval_path=eval_path,
        output_dir=output_dir,
        max_length=a.max_length,
        per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_eval_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        num_train_epochs=a.num_train_epochs,
        max_steps=a.max_steps,
        warmup_ratio=a.warmup_ratio,
        logging_steps=a.logging_steps,
        save_steps=a.save_steps,
        eval_steps=a.eval_steps,
        seed=a.seed,
        fp16=bool(a.fp16),
        bf16=bool(a.bf16),
        gradient_checkpointing=bool(a.gradient_checkpointing),
        load_in_8bit=bool(a.load_in_8bit),
        load_in_4bit=bool(a.load_in_4bit),
        bnb_4bit_compute_dtype=str(a.bnb_4bit_compute_dtype),
        device_map=(str(a.device_map) if a.device_map not in (None, "", "none", "None") else None),
        resume_from_checkpoint=(str(a.resume_from_checkpoint) if a.resume_from_checkpoint not in (None, "", "none", "None") else None),
        eval_only=bool(a.eval_only),
        adapter_path=(str(a.adapter_path) if a.adapter_path not in (None, "", "none", "None") else None),
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    seq_len = len(input_ids)
    if prompt_len >= seq_len - 1:
        prompt_len = seq_len - 2
    if prompt_len < 0:
        prompt_len = 0
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return labels


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    print(f"Starting training for domain: {cfg.domain}")
    print(f"Base model: {cfg.base_model_path}")
    print(f"Output directory: {cfg.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|pad|>"
    tokenizer.padding_side = "right"

    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Only one of --load_in_8bit / --load_in_4bit can be enabled.")

    torch_dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None)

    quantization_config = None
    if cfg.load_in_8bit or cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=cfg.device_map,
    )
    
    if getattr(model, "get_input_embeddings", None) and getattr(tokenizer, "vocab_size", None):
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"): model.config.use_cache = False

    if cfg.load_in_8bit or cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    if cfg.eval_only:
        model = PeftModel.from_pretrained(model, cfg.adapter_path or cfg.output_dir, is_trainable=False)
    else:
        lora_params = DOMAIN_LORA_CONFIG.get(cfg.domain, DOMAIN_LORA_CONFIG["default"])
        print(f"Using LoRA config for {cfg.domain}: r={lora_params['r']}, alpha={lora_params['alpha']}")
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['alpha'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    train_samples = load_instruction_dataset(cfg.train_path, keep_extra_fields=False)
    if cfg.max_train_samples: train_samples = train_samples[:cfg.max_train_samples]

    eval_samples = load_instruction_dataset(cfg.eval_path, keep_extra_fields=False) if cfg.eval_path else None
    if eval_samples and cfg.max_eval_samples: eval_samples = eval_samples[:cfg.max_eval_samples]

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, samples: list[Any]): self._samples = samples
        def __len__(self) -> int: return len(self._samples)
        def __getitem__(self, idx: int) -> dict[str, Any]:
            s = self._samples[idx]
            prompt = format_prompt(s)
            full_text = build_sft_text(s, eos_token=tokenizer.eos_token)
            enc_full = tokenizer(full_text, truncation=True, max_length=cfg.max_length)
            enc_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False)
            input_ids, prompt_len = enc_full["input_ids"], len(enc_prompt["input_ids"])
            labels = _build_labels(input_ids, prompt_len)
            if all(x == -100 for x in labels): labels[-1] = input_ids[-1]
            return {"input_ids": input_ids, "attention_mask": enc_full["attention_mask"], "labels": labels}

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            ids, mask, lab = x["input_ids"], x["attention_mask"], x["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append(mask + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in zip(["input_ids", "attention_mask", "labels"], [input_ids, attention_mask, labels])}

    train_dataset = SFTDataset(train_samples)
    eval_dataset = SFTDataset(eval_samples) if eval_samples else None

    max_steps = cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1
    num_train_epochs = 1.0 if max_steps > 0 else cfg.num_train_epochs

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate)

    if not cfg.eval_only:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        # ✅ 新增：保存 domain 信息
        meta_path = Path(cfg.output_dir) / "adapter_config.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"domain": cfg.domain}, f, ensure_ascii=False, indent=2)

    if eval_dataset:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        if isinstance(eval_loss, (float, int)) and eval_loss < 50:
            metrics["eval_ppl"] = math.exp(float(eval_loss))
        for k, v in sorted(metrics.items()):
            print(f"{k}={v}")


if __name__ == "__main__":
    main()

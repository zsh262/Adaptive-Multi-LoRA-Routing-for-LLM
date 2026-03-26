from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, default="models/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter_path", type=str, default="models/lora/coding_lora")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--device_map", type=str, default=None)

    p.add_argument("--dataset_path", type=str, default="data/coding/test.json")
    p.add_argument("--max_examples", type=int, default=10)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--do_sample", action="store_true")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
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


def _load_tokenizer_and_base_model(
    *,
    base_model_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    bnb_4bit_compute_dtype: str,
    device_map: str | None,
    max_length: int,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of --load_in_4bit / --load_in_8bit can be enabled.")

    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.float16 if bnb_4bit_compute_dtype == "float16" else torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.eval()
    if getattr(model.config, "max_position_embeddings", None) is not None:
        if max_length > int(model.config.max_position_embeddings):
            raise ValueError(
                f"max_length({max_length}) > model.max_position_embeddings({model.config.max_position_embeddings})"
            )
    return tokenizer, model


def _attach_lora(model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def _generate(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    gen: GenConfig,
    max_length: int,
) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
            top_k=gen.top_k,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    from training.dataset_loader import format_prompt, load_instruction_dataset

    tokenizer, base_model = _load_tokenizer_and_base_model(
        base_model_path=args.base_model_path,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        bnb_4bit_compute_dtype=str(args.bnb_4bit_compute_dtype),
        device_map=(str(args.device_map) if args.device_map not in (None, "", "none", "None") else None),
        max_length=int(args.max_length),
    )
    lora_model = _attach_lora(base_model, args.adapter_path)

    samples = load_instruction_dataset(Path(args.dataset_path), keep_extra_fields=True)
    samples = samples[: max(0, int(args.max_examples))]

    gen = GenConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        do_sample=bool(args.do_sample),
    )

    for i, s in enumerate(samples, start=1):
        prompt = format_prompt(s)
        base_out = _generate(tokenizer=tokenizer, model=base_model, prompt=prompt, gen=gen, max_length=args.max_length)
        lora_out = _generate(tokenizer=tokenizer, model=lora_model, prompt=prompt, gen=gen, max_length=args.max_length)

        print("=" * 88)
        print(f"[{i}/{len(samples)}] PROMPT")
        print(prompt)
        print("-" * 88)
        print("BASE")
        print(base_out)
        print("-" * 88)
        print("LORA")
        print(lora_out)
        print("-" * 88)
        print("REF")
        print(s.output)


if __name__ == "__main__":
    main()

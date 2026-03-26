from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PROMPT_TEMPLATE = "Instruction: {instruction}\nInput: {input}\nAnswer:"


@dataclass(frozen=True)
class InstructionSample:
    instruction: str
    input: str
    output: str
    meta: dict[str, Any]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def parse_instruction_sample(
    obj: Mapping[str, Any],
    *,
    keep_extra_fields: bool = True,
) -> InstructionSample:
    instruction = _as_str(obj.get("instruction"))
    input_text = _as_str(obj.get("input"))
    output = _as_str(obj.get("output"))

    if keep_extra_fields:
        meta = {k: v for k, v in obj.items() if k not in {"instruction", "input", "output"}}
    else:
        meta = {}

    return InstructionSample(
        instruction=instruction,
        input=input_text,
        output=output,
        meta=meta,
    )


def format_prompt(
    sample: InstructionSample,
    *,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    return template.format(
        instruction=sample.instruction,
        input=sample.input,
    )


def build_sft_text(
    sample: InstructionSample,
    *,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    eos_token: str | None = None,
) -> str:
    prompt = format_prompt(sample, template=template)
    text = f"{prompt}\n{sample.output}"
    if eos_token:
        text += eos_token
    return text


def load_instruction_dataset(
    path: str | Path,
    *,
    keep_extra_fields: bool = True,
    encoding: str = "utf-8",
) -> list[InstructionSample]:
    p = Path(path)
    with p.open("r", encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {p.as_posix()}, got {type(data).__name__}")

    samples: list[InstructionSample] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Expected each item to be an object in {p.as_posix()}, got {type(item).__name__}"
            )
        samples.append(parse_instruction_sample(item, keep_extra_fields=keep_extra_fields))

    return samples


def iter_instruction_dataset(
    items: Iterable[Mapping[str, Any]],
    *,
    keep_extra_fields: bool = True,
) -> Iterable[InstructionSample]:
    for item in items:
        yield parse_instruction_sample(item, keep_extra_fields=keep_extra_fields)

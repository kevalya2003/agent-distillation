"""Generate held-out first-tool decisions from a base model or LoRA adapter."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_distillation.metrics import parse_tool_call
from agent_distillation.train import load_experiment_config


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _prompt_before_first_tool(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Reference record has no messages list")
    prefix: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Message must be an object")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant" and parse_tool_call(content) is not None:
            return prefix
        prefix.append({"role": role, "content": content})
    raise ValueError("Reference has no assistant tool call")


def generate_predictions(
    config_path: Path,
    reference_path: Path,
    output_path: Path,
    adapter_path: Optional[Path] = None,
) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            'Generation dependencies are missing. Install with pip install -e ".[train]"'
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit generation requires an NVIDIA CUDA GPU")
    config = load_experiment_config(config_path)
    model_config = config["model"]
    quantization = config["quantization"]
    generation = config.get("generation", {})
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            quantization.get("use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_name = str(model_config["name"])
    tokenizer_source = str(adapter_path) if adapter_path is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    records = _load_records(reference_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt_messages = _prompt_before_first_tool(record)
            input_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=int(generation.get("max_new_tokens", 256)),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(
                generated[0, input_ids.shape[-1] :],
                skip_special_tokens=True,
            )
            handle.write(
                json.dumps(
                    {
                        "trajectory_id": str(record["trajectory_id"]),
                        "prediction": prediction,
                        "model": model_name,
                        "adapter": str(adapter_path) if adapter_path else None,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
            handle.flush()


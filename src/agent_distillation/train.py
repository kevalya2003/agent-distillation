"""QLoRA supervised fine-tuning entry point.

Heavy GPU libraries are imported lazily so data preparation remains lightweight.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_experiment_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    for section in (
        "model",
        "data",
        "output",
        "quantization",
        "lora",
        "training",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError("Missing configuration section: {}".format(section))
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in (
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def train(config_path: Path) -> Dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            'Training dependencies are missing. Install with pip install -e ".[train]"'
        ) from error

    config = load_experiment_config(config_path)
    model_config = config["model"]
    data_config = config["data"]
    output_config = config["output"]
    quantization_config = config["quantization"]
    lora_config = config["lora"]
    training_config = config["training"]

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires an NVIDIA CUDA GPU")
    set_seed(int(training_config["seed"]))

    dtype_name = str(quantization_config.get("compute_dtype", "auto"))
    if dtype_name == "auto":
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    elif dtype_name == "bfloat16":
        compute_dtype = torch.bfloat16
    elif dtype_name == "float16":
        compute_dtype = torch.float16
    else:
        raise ValueError("compute_dtype must be auto, bfloat16, or float16")

    bitsandbytes = BitsAndBytesConfig(
        load_in_4bit=bool(quantization_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quantization_config.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            quantization_config.get("use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_name = str(model_config["name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        quantization_config=bitsandbytes,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(
            training_config.get("gradient_checkpointing", True)
        ),
    )
    model.config.use_cache = False

    train_file = Path(str(data_config["train_file"]))
    eval_file = Path(str(data_config["eval_file"]))
    if not train_file.is_file() or not eval_file.is_file():
        raise FileNotFoundError(
            "Run agent-distill prepare before training; processed data is missing"
        )
    datasets = load_dataset(
        "json",
        data_files={"train": str(train_file), "eval": str(eval_file)},
    )
    if len(datasets["train"]) == 0 or len(datasets["eval"]) == 0:
        raise ValueError("Both train and eval splits must contain examples")

    output_directory = Path(str(output_config["directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(output_directory),
        num_train_epochs=float(training_config["epochs"]),
        learning_rate=float(training_config["learning_rate"]),
        per_device_train_batch_size=int(
            training_config["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        gradient_checkpointing=bool(
            training_config.get("gradient_checkpointing", True)
        ),
        warmup_ratio=float(training_config["warmup_ratio"]),
        max_length=int(training_config["max_length"]),
        logging_steps=int(training_config["logging_steps"]),
        save_strategy=str(training_config["save_strategy"]),
        eval_strategy=str(training_config["eval_strategy"]),
        report_to=str(training_config.get("report_to", "none")),
        seed=int(training_config["seed"]),
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        eos_token=training_config.get("eos_token"),
        packing=False,
    )
    peft_config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=lora_config.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(str(output_directory))

    manifest: Dict[str, Any] = {
        "config_path": str(config_path),
        "config": config,
        "package_versions": _package_versions(),
        "data": {
            "train_file": str(train_file),
            "train_sha256": _sha256(train_file),
            "train_examples": len(datasets["train"]),
            "eval_file": str(eval_file),
            "eval_sha256": _sha256(eval_file),
            "eval_examples": len(datasets["eval"]),
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": evaluation_metrics,
    }
    (output_directory / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest


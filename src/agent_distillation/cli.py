"""Command-line interface for preparation, training, generation, and scoring."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from agent_distillation.generate import generate_predictions
from agent_distillation.metrics import score_predictions
from agent_distillation.prepare import load_verified_ids, prepare_dataset
from agent_distillation.train import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qwen25_coder_3b_qlora.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-distill",
        description="Curate and distill successful agent tool-use trajectories.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, action="append", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--eval-ratio", type=float, default=0.2)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--max-steps", type=int, default=30)
    prepare_parser.add_argument("--max-tool-output-chars", type=int, default=4_000)
    prepare_parser.add_argument(
        "--verified-ids",
        type=Path,
        help="Optional JSON list or newline file of externally passing task IDs",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser.add_argument("--adapter", type=Path)
    generate_parser.add_argument("--references", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--references", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.subcommand == "prepare":
        report = prepare_dataset(
            inputs=arguments.input,
            output_directory=arguments.output,
            eval_ratio=arguments.eval_ratio,
            seed=arguments.seed,
            max_steps=arguments.max_steps,
            max_tool_output_chars=arguments.max_tool_output_chars,
            verified_ids=(
                load_verified_ids(arguments.verified_ids)
                if arguments.verified_ids is not None
                else None
            ),
        )
        print(json.dumps(report, indent=2))
        return 0

    if arguments.subcommand == "train":
        manifest = train(arguments.config)
        print(json.dumps(manifest["eval_metrics"], indent=2, default=str))
        return 0

    if arguments.subcommand == "generate":
        generate_predictions(
            config_path=arguments.config,
            reference_path=arguments.references,
            output_path=arguments.output,
            adapter_path=arguments.adapter,
        )
        print("Wrote predictions to {}".format(arguments.output))
        return 0

    metrics = score_predictions(arguments.references, arguments.predictions)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Rejection sampling and leakage-resistant train/evaluation splitting."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from agent_distillation.trajectory import (
    TrajectoryError,
    build_messages,
    has_required_actions,
    load_trajectory,
)


def discover_trajectories(inputs: Sequence[Path]) -> List[Path]:
    discovered = set()
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".jsonl":
            discovered.add(path.resolve())
        elif path.is_dir():
            discovered.update(item.resolve() for item in path.rglob("*.jsonl"))
        else:
            raise ValueError("Input is not a JSONL file or directory: {}".format(path))
    return sorted(discovered, key=str)


def load_verified_ids(path: Path) -> Set[str]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        identifiers = {str(item).strip() for item in value if str(item).strip()}
    else:
        identifiers = {
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    if not identifiers:
        raise ValueError("Verified ID file is empty: {}".format(path))
    return identifiers


def _conversation_hash(messages: List[Dict[str, str]]) -> str:
    canonical = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _group_split(
    records: Sequence[Dict[str, Any]], eval_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata", {})
        group = (
            metadata.get("task_group")
            if isinstance(metadata, dict)
            else None
        )
        groups[str(group or record["trajectory_id"])].append(record)
    if eval_ratio == 0.0 or len(groups) < 2:
        return list(records), []

    ordered_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            "{}:{}".format(seed, group).encode("utf-8")
        ).hexdigest(),
    )
    eval_group_count = max(1, round(len(ordered_groups) * eval_ratio))
    eval_group_count = min(eval_group_count, len(ordered_groups) - 1)
    eval_groups = set(ordered_groups[:eval_group_count])
    train = [
        record
        for record in records
        if str(record["metadata"].get("task_group") or record["trajectory_id"])
        not in eval_groups
    ]
    evaluation = [
        record
        for record in records
        if str(record["metadata"].get("task_group") or record["trajectory_id"])
        in eval_groups
    ]
    return train, evaluation


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def prepare_dataset(
    inputs: Sequence[Path],
    output_directory: Path,
    eval_ratio: float = 0.2,
    seed: int = 42,
    max_steps: int = 30,
    max_tool_output_chars: int = 4_000,
    verified_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    paths = discover_trajectories(inputs)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rejection_reasons: Counter = Counter()
    tool_frequencies: Counter = Counter()
    records: List[Dict[str, Any]] = []
    seen_conversations = set()

    for path in paths:
        try:
            trajectory = load_trajectory(path)
        except (OSError, TrajectoryError, ValueError):
            rejection_reasons["invalid"] += 1
            continue
        if not trajectory.success:
            rejection_reasons["unsuccessful"] += 1
            continue
        if verified_ids is not None and trajectory.trajectory_id not in verified_ids:
            rejection_reasons["not_externally_verified"] += 1
            continue
        if trajectory.steps > max_steps:
            rejection_reasons["too_many_steps"] += 1
            continue
        if not has_required_actions(trajectory):
            rejection_reasons["missing_required_action"] += 1
            continue

        messages = build_messages(trajectory, max_tool_output_chars)
        digest = _conversation_hash(messages)
        if digest in seen_conversations:
            rejection_reasons["duplicate"] += 1
            continue
        seen_conversations.add(digest)
        tool_frequencies.update(trajectory.tool_names)
        task_group = hashlib.sha256(
            " ".join(trajectory.issue.lower().split()).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "messages": messages,
                "metadata": {
                    "source_path": trajectory.source_path,
                    "steps": trajectory.steps,
                    "status": trajectory.status,
                    "tool_names": trajectory.tool_names,
                    "conversation_sha256": digest,
                    "task_group": task_group,
                },
            }
        )

    train, evaluation = _group_split(records, eval_ratio, seed)
    train.sort(key=lambda record: str(record["trajectory_id"]))
    evaluation.sort(key=lambda record: str(record["trajectory_id"]))
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "eval.jsonl", evaluation)

    accepted_steps = [int(record["metadata"]["steps"]) for record in records]
    report: Dict[str, Any] = {
        "discovered": len(paths),
        "accepted": len(records),
        "rejected": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "eval_ratio": eval_ratio,
        "seed": seed,
        "max_steps": max_steps,
        "external_verification_filter": verified_ids is not None,
        "average_accepted_steps": (
            sum(accepted_steps) / len(accepted_steps) if accepted_steps else 0.0
        ),
        "tool_frequencies": dict(sorted(tool_frequencies.items())),
        "warning": (
            "At least two accepted trajectory IDs are required for a held-out split."
            if records and not evaluation
            else None
        ),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


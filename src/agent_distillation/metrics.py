"""Held-out structured tool-call metrics."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = _TOOL_CALL.search(text)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("name"), str)
        or not isinstance(value.get("arguments"), dict)
    ):
        return None
    return {"name": value["name"], "arguments": value["arguments"]}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} must be an object".format(path, line_number))
            records.append(value)
    return records


def _reference_call(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            parsed = parse_tool_call(str(message.get("content", "")))
            if parsed is not None:
                return parsed
    return None


def _argument_key_f1(
    reference: Dict[str, Any], prediction: Dict[str, Any]
) -> float:
    expected = set(reference.get("arguments", {}).keys())
    actual = set(prediction.get("arguments", {}).keys())
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_predictions(
    reference_path: Path, prediction_path: Path
) -> Dict[str, Any]:
    references: Dict[str, Dict[str, Any]] = {}
    for record in _read_jsonl(reference_path):
        trajectory_id = str(record["trajectory_id"])
        call = _reference_call(record)
        if call is not None:
            references[trajectory_id] = call

    predictions: Dict[str, str] = {}
    for record in _read_jsonl(prediction_path):
        trajectory_id = str(record["trajectory_id"])
        if trajectory_id in predictions:
            raise ValueError("Duplicate prediction: {}".format(trajectory_id))
        predictions[trajectory_id] = str(record.get("prediction", ""))

    parsed_count = 0
    name_correct = 0
    exact_correct = 0
    argument_f1_total = 0.0
    for trajectory_id, reference in references.items():
        prediction = parse_tool_call(predictions.get(trajectory_id, ""))
        if prediction is None:
            continue
        parsed_count += 1
        if prediction["name"] == reference["name"]:
            name_correct += 1
        if prediction == reference:
            exact_correct += 1
        argument_f1_total += _argument_key_f1(reference, prediction)

    total = len(references)
    return {
        "examples": total,
        "predictions_found": sum(
            1 for trajectory_id in references if trajectory_id in predictions
        ),
        "parse_rate": parsed_count / total if total else 0.0,
        "tool_name_accuracy": name_correct / total if total else 0.0,
        "argument_key_f1": argument_f1_total / total if total else 0.0,
        "exact_call_accuracy": exact_correct / total if total else 0.0,
    }


import json
from pathlib import Path

from agent_distillation.metrics import parse_tool_call, score_predictions


def write_jsonl(path: Path, records: list) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_parse_tool_call_requires_valid_shape() -> None:
    assert parse_tool_call(
        '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
    ) == {"name": "read_file", "arguments": {"path": "a.py"}}
    assert parse_tool_call("<tool_call>not-json</tool_call>") is None
    assert parse_tool_call('{"name":"read_file","arguments":{}}') is None


def test_score_predictions_counts_missing_and_malformed_calls(tmp_path: Path) -> None:
    references = tmp_path / "eval.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        references,
        [
            {
                "trajectory_id": "one",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"read_file",'
                            '"arguments":{"path":"a.py"}}</tool_call>'
                        ),
                    }
                ],
            },
            {
                "trajectory_id": "two",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"run_tests",'
                            '"arguments":{}}</tool_call>'
                        ),
                    }
                ],
            },
        ],
    )
    write_jsonl(
        predictions,
        [
            {
                "trajectory_id": "one",
                "prediction": (
                    '<tool_call>{"name":"read_file",'
                    '"arguments":{"path":"a.py"}}</tool_call>'
                ),
            },
            {"trajectory_id": "two", "prediction": "malformed"},
        ],
    )

    metrics = score_predictions(references, predictions)

    assert metrics["parse_rate"] == 0.5
    assert metrics["tool_name_accuracy"] == 0.5
    assert metrics["argument_key_f1"] == 0.5
    assert metrics["exact_call_accuracy"] == 0.5


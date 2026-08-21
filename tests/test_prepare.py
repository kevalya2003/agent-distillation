import json
from pathlib import Path

from agent_distillation.prepare import load_verified_ids, prepare_dataset


def events(issue: str, success: bool = True, include_submit: bool = True) -> list:
    values = [
        {"type": "run_started", "issue": issue},
        {
            "type": "model_response",
            "content": "",
            "tool_calls": [{"id": "1", "name": "run_tests", "arguments": {}}],
        },
        {"type": "tool_call", "tool": "run_tests", "arguments": {}},
        {
            "type": "tool_result",
            "tool": "run_tests",
            "ok": True,
            "content": "passed",
            "terminal": False,
        },
    ]
    if include_submit:
        values.extend(
            [
                {
                    "type": "model_response",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "2",
                            "name": "submit",
                            "arguments": {"summary": issue},
                        }
                    ],
                },
                {
                    "type": "tool_call",
                    "tool": "submit",
                    "arguments": {"summary": issue},
                },
            ]
        )
    values.append(
        {
            "type": "run_finished",
            "result": {
                "success": success,
                "status": "submitted" if success else "failed",
                "steps": 2,
            },
        }
    )
    return values


def write_trajectory(path: Path, values: list) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def read_ids(path: Path) -> set:
    return {
        json.loads(line)["trajectory_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def test_preparation_filters_and_splits_by_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_trajectory(source / "task-a.jsonl", events("Issue A"))
    write_trajectory(source / "task-b.jsonl", events("Issue B"))
    write_trajectory(source / "failed.jsonl", events("Issue C", success=False))
    write_trajectory(
        source / "incomplete.jsonl", events("Issue D", include_submit=False)
    )
    output = tmp_path / "processed"

    report = prepare_dataset([source], output, eval_ratio=0.5, seed=7)

    train_ids = read_ids(output / "train.jsonl")
    eval_ids = read_ids(output / "eval.jsonl")
    assert report["accepted"] == 2
    assert report["rejection_reasons"] == {
        "missing_required_action": 1,
        "unsuccessful": 1,
    }
    assert len(train_ids) == 1
    assert len(eval_ids) == 1
    assert train_ids.isdisjoint(eval_ids)


def test_external_verification_filter_accepts_only_listed_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_trajectory(source / "task-a.jsonl", events("Issue A"))
    write_trajectory(source / "task-b.jsonl", events("Issue B"))
    verified_path = tmp_path / "verified.json"
    verified_path.write_text('["task-b"]', encoding="utf-8")

    report = prepare_dataset(
        [source],
        tmp_path / "processed",
        eval_ratio=0.0,
        verified_ids=load_verified_ids(verified_path),
    )

    assert report["accepted"] == 1
    assert report["rejection_reasons"] == {"not_externally_verified": 1}


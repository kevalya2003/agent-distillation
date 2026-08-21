import json
from pathlib import Path

import pytest

from agent_distillation.trajectory import (
    TrajectoryError,
    build_messages,
    has_required_actions,
    load_trajectory,
)


def write_events(path: Path, events: list) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def successful_events(issue: str = "Fix the bug.") -> list:
    return [
        {"type": "run_started", "issue": issue},
        {
            "type": "model_response",
            "content": "",
            "tool_calls": [
                {"id": "1", "name": "read_file", "arguments": {"path": "app.py"}}
            ],
        },
        {"type": "tool_call", "tool": "read_file", "arguments": {"path": "app.py"}},
        {
            "type": "tool_result",
            "tool": "read_file",
            "ok": True,
            "content": "x" * 100,
            "terminal": False,
        },
        {
            "type": "model_response",
            "content": "",
            "tool_calls": [{"id": "2", "name": "run_tests", "arguments": {}}],
        },
        {"type": "tool_call", "tool": "run_tests", "arguments": {}},
        {
            "type": "tool_result",
            "tool": "run_tests",
            "ok": True,
            "content": "passed",
            "terminal": False,
        },
        {
            "type": "model_response",
            "content": "",
            "tool_calls": [
                {
                    "id": "3",
                    "name": "submit",
                    "arguments": {"summary": "fixed"},
                }
            ],
        },
        {"type": "tool_call", "tool": "submit", "arguments": {"summary": "fixed"}},
        {
            "type": "tool_result",
            "tool": "submit",
            "ok": True,
            "content": "accepted",
            "terminal": True,
        },
        {
            "type": "run_finished",
            "result": {"success": True, "status": "submitted", "steps": 3},
        },
    ]


def test_load_and_convert_successful_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "task-1.jsonl"
    write_events(path, successful_events())

    trajectory = load_trajectory(path)
    messages = build_messages(trajectory, max_tool_output_chars=20)

    assert trajectory.success is True
    assert has_required_actions(trajectory) is True
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Fix the bug."}
    assert '"name":"read_file"' in messages[2]["content"]
    assert "output truncated" in messages[3]["content"]
    assert all("accepted" not in message["content"] for message in messages)


def test_incomplete_trajectory_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    write_events(path, [{"type": "run_started", "issue": "broken"}])

    with pytest.raises(TrajectoryError, match="run boundaries"):
        load_trajectory(path)


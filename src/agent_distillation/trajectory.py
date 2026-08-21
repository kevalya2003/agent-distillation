"""Parsing and normalization of coding-agent JSONL trajectories."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


DISTILLATION_SYSTEM_PROMPT = """You are a coding agent working in one repository.
Inspect relevant files, make the smallest correct change, run verification after the final
edit, and finish with submit. Tool calls must use this exact form:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>
"""


class TrajectoryError(ValueError):
    """The JSONL file is incomplete or does not match the expected schema."""


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    source_path: str
    issue: str
    success: bool
    status: str
    steps: int
    events: List[Dict[str, Any]]

    @property
    def tool_names(self) -> List[str]:
        names: List[str] = []
        for event in self.events:
            if event.get("type") == "tool_call":
                names.append(str(event.get("tool", "")))
        return names


def load_trajectory(path: Path) -> Trajectory:
    source = Path(path)
    events: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrajectoryError(
                    "{}:{} contains invalid JSON".format(source, line_number)
                ) from error
            if not isinstance(event, dict) or "type" not in event:
                raise TrajectoryError(
                    "{}:{} is not a trajectory event".format(source, line_number)
                )
            events.append(event)

    if not events:
        raise TrajectoryError("{} is empty".format(source))
    started = next(
        (event for event in events if event.get("type") == "run_started"), None
    )
    finished = next(
        (event for event in reversed(events) if event.get("type") == "run_finished"),
        None,
    )
    if started is None or finished is None:
        raise TrajectoryError("{} is missing run boundaries".format(source))
    issue = str(started.get("issue", "")).strip()
    result = finished.get("result")
    if not issue or not isinstance(result, dict):
        raise TrajectoryError("{} has invalid run metadata".format(source))

    return Trajectory(
        trajectory_id=source.stem,
        source_path=str(source),
        issue=issue,
        success=bool(result.get("success", False)),
        status=str(result.get("status", "unknown")),
        steps=int(result.get("steps", 0)),
        events=events,
    )


def build_messages(
    trajectory: Trajectory, max_tool_output_chars: int = 4_000
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
        {"role": "user", "content": trajectory.issue},
    ]
    for event in trajectory.events:
        event_type = event.get("type")
        if event_type == "model_response":
            parts: List[str] = []
            content = str(event.get("content", "")).strip()
            if content:
                parts.append(content)
            raw_calls = event.get("tool_calls", [])
            if not isinstance(raw_calls, list):
                raise TrajectoryError("model_response.tool_calls must be a list")
            for call in raw_calls:
                if not isinstance(call, dict):
                    raise TrajectoryError("tool call must be an object")
                encoded = json.dumps(
                    {
                        "name": str(call.get("name", "")),
                        "arguments": call.get("arguments", {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                parts.append("<tool_call>{}</tool_call>".format(encoded))
            if parts:
                messages.append({"role": "assistant", "content": "\n".join(parts)})
        elif event_type == "tool_result" and not bool(event.get("terminal", False)):
            tool_content = str(event.get("content", ""))
            if len(tool_content) > max_tool_output_chars:
                tool_content = (
                    tool_content[:max_tool_output_chars] + "\n... output truncated ..."
                )
            encoded_result = json.dumps(
                {
                    "name": str(event.get("tool", "")),
                    "ok": bool(event.get("ok", False)),
                    "content": tool_content,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            messages.append(
                {
                    "role": "tool",
                    "content": "<tool_result>{}</tool_result>".format(encoded_result),
                }
            )
    return messages


def has_required_actions(
    trajectory: Trajectory, required_actions: Sequence[str] = ("run_tests", "submit")
) -> bool:
    names = set(trajectory.tool_names)
    return all(action in names for action in required_actions)


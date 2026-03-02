from __future__ import annotations

from thenvoi.adapters.codex.events import (
    build_task_event_content,
    task_event_id,
    task_event_summary,
    task_event_title,
)


def test_task_event_extractors_handle_flat_and_nested_shapes() -> None:
    params = {"taskId": "t-flat", "title": "Compile", "summary": "Started"}
    assert task_event_id(params) == "t-flat"
    assert task_event_title(params) == "Compile"
    assert task_event_summary(params) == "Started"

    nested = {"task": {"id": "t-nested", "name": "Ship", "description": "Deploy"}}
    assert task_event_id(nested) == "t-nested"
    assert task_event_title(nested) == "Ship"
    assert task_event_summary(nested) == "Deploy"


def test_build_task_event_content_omits_redundant_summary() -> None:
    content = build_task_event_content(
        task_id="t1",
        task="Index",
        status="completed",
        summary="Index",
    )
    assert "UUID: t1" in content
    assert "Task: Index" in content
    assert "Status: completed" in content
    assert "Summary:" not in content


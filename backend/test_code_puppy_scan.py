"""Tests for Code Puppy session scanning.

Code Puppy now exposes per-session transcript logs under:
  ~/.code_puppy/logs/session_transcripts/<session_id>.jsonl
or, when XDG_STATE_HOME is explicitly set:
  $XDG_STATE_HOME/code_puppy/logs/session_transcripts/<session_id>.jsonl

TokenTelemetry should prefer these structured session transcripts for first-class
session support, while retaining the older model_usage.jsonl fallback.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import main  # noqa: E402
from pricing import calculate_cost  # noqa: E402


def _write_usage_log(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _write_transcript(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_scan_code_puppy_prefers_session_transcripts(tmp_path, monkeypatch):
    transcript = _write_transcript(tmp_path / "logs" / "session_transcripts" / "sess-one.jsonl", [
        {
            "event": "user_prompt",
            "logged_at": "2026-07-17T05:10:17.215286+00:00",
            "role": "user",
            "sequence": 1,
            "session_id": "sess-one",
            "text": "Update the team links",
            "version": 1,
        },
        {
            "agent_name": "planning-agent",
            "event": "run_start",
            "logged_at": "2026-07-17T05:10:17.216561+00:00",
            "model_name": "claude-code-claude-opus-4-7",
            "sequence": 2,
            "session_id": "sess-one",
            "version": 1,
        },
        {
            "event": "tool_start",
            "logged_at": "2026-07-17T05:10:18.000000+00:00",
            "sequence": 3,
            "session_id": "sess-one",
            "tool_name": "read_file",
            "tool_args": {"path": "src/app.js"},
            "version": 1,
        },
        {
            "agent_name": "review-agent",
            "error": None,
            "event": "run_end",
            "logged_at": "2026-07-17T05:11:21.456794+00:00",
            "metadata": {
                "model": "gpt-5",
                "usage_cached_read_tokens": 25,
                "usage_cached_write_tokens": 5,
                "usage_input_tokens": 250,
                "usage_output_tokens": 50,
                "usage_total_tokens": 300,
            },
            "model_name": "gpt-5",
            "response_text": "Done.",
            "sequence": 4,
            "session_id": "sess-one",
            "success": True,
            "version": 1,
        },
        {
            "agent_name": "planning-agent",
            "error": "boom",
            "event": "run_end",
            "logged_at": "2026-07-17T05:12:21.456794+00:00",
            "metadata": {
                "model": "claude-code-claude-opus-4-7",
                "usage_cached_read_tokens": 300,
                "usage_cached_write_tokens": 100,
                "usage_input_tokens": 1000,
                "usage_output_tokens": 200,
                "usage_total_tokens": 1200,
            },
            "model_name": "claude-code-claude-opus-4-7",
            "response_text": "Failed after trying.",
            "sequence": 5,
            "session_id": "sess-one",
            "success": False,
            "version": 1,
        },
    ])
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", transcript.parent)
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", tmp_path / "logs" / "model_usage.jsonl")

    out = main._scan_code_puppy_sessions()
    assert len(out) == 1
    s = out[0]
    assert s["agent"] == "code_puppy"
    assert s["id"] == "sess-one"
    assert s["project"] == "unknown"
    assert s["model"] == "claude-code-claude-opus-4-7"
    assert s["display"] == "Update the team links"
    assert s["mcp_tools"] == ["read_file"]

    tk = s["tokens"]
    assert tk["input"] == 600 + 220
    assert tk["output"] == 200 + 50
    assert tk["cached"] == 300 + 25
    assert tk["cache_creation"] == 100 + 5
    assert tk["total"] == tk["input"] + tk["output"] + tk["cached"] + tk["cache_creation"]

    expected = (
        calculate_cost("claude-code-claude-opus-4-7", 600, 200, 300, cache_creation_tokens=100)
        + calculate_cost("gpt-5", 220, 50, 25, cache_creation_tokens=5)
    )
    assert s["cost"] == pytest.approx(expected)
    assert tk["cost"] == pytest.approx(expected)

    meta = s["code_puppy"]
    assert meta["agents_used"] == ["planning-agent", "review-agent"]
    assert meta["models_used"] == ["claude-code-claude-opus-4-7", "gpt-5"]
    assert meta["successful_calls"] == 1
    assert meta["failed_calls"] == 1
    assert meta["response_chars"] == len("Done.") + len("Failed after trying.")
    assert meta["transcript_events"] == 5
    assert meta["source"] == "session_transcript"


def test_scan_code_puppy_missing_log_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", tmp_path / "missing.jsonl")
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", tmp_path / "session_transcripts")
    assert main._scan_code_puppy_sessions() == []


def test_code_puppy_detected_when_usage_log_exists(tmp_path, monkeypatch):
    log = _write_usage_log(tmp_path / "logs" / "model_usage.jsonl", [])
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", log)
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", tmp_path / "logs" / "session_transcripts")
    assert "code_puppy" in main._list_available_agents()

    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", tmp_path / "gone.jsonl")
    assert "code_puppy" not in main._list_available_agents()


def test_code_puppy_detected_when_transcript_exists(tmp_path, monkeypatch):
    transcript = _write_transcript(tmp_path / "logs" / "session_transcripts" / "sess-det.jsonl", [
        {
            "event": "user_prompt",
            "logged_at": "2026-07-17T05:10:17.215286+00:00",
            "role": "user",
            "sequence": 1,
            "session_id": "sess-det",
            "text": "hello",
            "version": 1,
        },
    ])
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", tmp_path / "gone.jsonl")
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", transcript.parent)
    assert "code_puppy" in main._list_available_agents()


def test_session_detail_code_puppy_normalizes_transcript_steps(tmp_path, monkeypatch):
    transcript = _write_transcript(tmp_path / "logs" / "session_transcripts" / "sess-detail.jsonl", [
        {
            "event": "user_prompt",
            "logged_at": "2026-07-17T05:10:17.215286+00:00",
            "role": "user",
            "sequence": 1,
            "session_id": "sess-detail",
            "text": "Please fix it",
            "version": 1,
        },
        {
            "event": "tool_start",
            "logged_at": "2026-07-17T05:10:18.000000+00:00",
            "sequence": 2,
            "session_id": "sess-detail",
            "tool_name": "read_file",
            "tool_args": {"path": "math.py"},
            "version": 1,
        },
        {
            "event": "tool_end",
            "logged_at": "2026-07-17T05:10:19.000000+00:00",
            "sequence": 3,
            "session_id": "sess-detail",
            "tool_name": "read_file",
            "result_summary": {"text": "contents"},
            "success": True,
            "version": 1,
        },
        {
            "agent_name": "planning-agent",
            "error": None,
            "event": "run_end",
            "logged_at": "2026-07-17T05:11:21.456794+00:00",
            "metadata": {
                "model": "claude-code-claude-opus-4-7",
                "usage_cached_read_tokens": 300,
                "usage_cached_write_tokens": 100,
                "usage_input_tokens": 1000,
                "usage_output_tokens": 200,
                "usage_total_tokens": 1200,
            },
            "model_name": "claude-code-claude-opus-4-7",
            "response_text": "All fixed.",
            "sequence": 4,
            "session_id": "sess-detail",
            "success": True,
            "version": 1,
        },
    ])
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", transcript.parent)
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", tmp_path / "logs" / "model_usage.jsonl")

    events = asyncio.run(main.get_session_detail("sess-detail", "code_puppy"))
    assert isinstance(events, list) and len(events) == 4
    assert events[0]["type"] == "user"
    assert events[0]["message"]["role"] == "user"
    assert events[1]["type"] == "tool_call"
    assert events[1]["payload"]["tool"] == "read_file"
    assert events[2]["type"] == "tool_result"
    assert events[3]["type"] == "assistant"
    assert events[3]["message"]["role"] == "assistant"
    assert events[3]["message"]["content"][0]["text"] == "All fixed."
    assert events[3]["message"]["usage"] == {
        "input_tokens": 600,
        "output_tokens": 200,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 100,
    }
    assert all("normalized_timestamp" in evt for evt in events)


def test_session_detail_code_puppy_usage_log_fallback(tmp_path, monkeypatch):
    log = _write_usage_log(tmp_path / "logs" / "model_usage.jsonl", [
        {
            "agent_name": "planning-agent",
            "model": "claude-code-claude-opus-4-7",
            "response_chars": 500,
            "session_id": "sess-usage",
            "success": True,
            "timestamp": "2026-07-16T12:34:56+00:00",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cached_read_tokens": 300,
                "cached_write_tokens": 100,
                "total_tokens": 1200,
            },
        },
    ])
    monkeypatch.setattr(main, "CODE_PUPPY_SESSION_TRANSCRIPTS_DIR", tmp_path / "logs" / "session_transcripts")
    monkeypatch.setattr(main, "CODE_PUPPY_MODEL_USAGE_LOG", log)

    events = asyncio.run(main.get_session_detail("sess-usage", "code_puppy"))
    assert isinstance(events, list) and len(events) == 1
    evt = events[0]
    assert evt["type"] == "assistant"
    assert evt["message"]["role"] == "assistant"
    assert "planning-agent" in evt["message"]["content"][0]["text"]
    assert evt["message"]["usage"] == {
        "input_tokens": 600,
        "output_tokens": 200,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 100,
    }
    assert evt["normalized_timestamp"] is not None

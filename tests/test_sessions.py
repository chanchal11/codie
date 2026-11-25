"""Conversation files under ~/.codie/conversations."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from codie.agent import CodingAgent
from codie.approvals import AllowAll
from codie.sessions import SessionRecorder, list_sessions, load_session
from codie.store import conversations_dir
from tests.support import ScriptedModel


def test_a_turn_round_trips_through_the_recorder(tmp_path: Path):
    model = ScriptedModel([AIMessage(content="hello back")])
    agent = CodingAgent(model, tmp_path, AllowAll())
    recorder = SessionRecorder(tmp_path)
    recorder.provider = "ollama"
    recorder.model = "scripted"
    agent.recorder = recorder
    assert agent.run_turn("hello there") == "hello back"
    found, errors = list_sessions(tmp_path)
    assert errors == []
    assert found[0].title == "hello there"
    assert found[0].model == "scripted"
    loaded = load_session(tmp_path, found[0].session_id)
    assert not isinstance(loaded, str)
    assert isinstance(loaded.messages[0], HumanMessage)
    assert loaded.messages[0].content == "hello there"
    assert loaded.messages[1].content == "hello back"


def test_a_corrupt_file_is_skipped(tmp_path: Path):
    root = conversations_dir(tmp_path)
    (root / "bad.json").write_text("{", encoding="utf-8")
    (root / "good.json").write_text(
        json.dumps(
            {
                "id": "good",
                "title": "kept",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "mode": "agent",
                "provider": "ollama",
                "model": "scripted",
                "search_provider": "ddgs",
                "messages": [{"type": "human", "content": "hi"}],
            }
        ),
        encoding="utf-8",
    )
    found, errors = list_sessions(tmp_path)
    assert [item.title for item in found] == ["kept"]
    assert errors == ["bad.json"]

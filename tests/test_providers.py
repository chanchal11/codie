"""Provider factory tests. They construct clients and do not call the models."""

import io
import json

import pytest

from codie.providers import bedrock_model_id, build_chat_model, list_ollama_models


def test_ollama_uses_the_requested_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    model = build_chat_model("ollama", "qwen2.5-coder")
    assert model.model == "qwen2.5-coder"


def test_ollama_model_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
    model = build_chat_model("ollama", None)
    assert model.model == "llama3.1"


def test_bedrock_requires_a_model_id(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="BEDROCK_MODEL_ID"):
        build_chat_model("bedrock", None)


def test_bedrock_constructs_from_env(monkeypatch):
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    model = build_chat_model("bedrock", None)
    assert model.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert bedrock_model_id() == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_unknown_provider():
    with pytest.raises(RuntimeError, match="Unknown provider"):
        build_chat_model("openai", "gpt")


def test_list_ollama_models(monkeypatch):
    payload = json.dumps({"models": [{"name": "qwen2.5-coder:latest"}, {"name": "llama3.1"}]}).encode()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(_url, timeout=0):
        assert timeout == 2
        return FakeResponse(payload)

    monkeypatch.setattr("codie.providers.urllib.request.urlopen", fake_urlopen)
    assert list_ollama_models("http://127.0.0.1:11434") == [
        "qwen2.5-coder:latest",
        "llama3.1",
    ]


def test_list_ollama_models_is_empty_when_daemon_is_down(monkeypatch):
    def fake_urlopen(_url, timeout=0):
        raise OSError("connection refused")

    monkeypatch.setattr("codie.providers.urllib.request.urlopen", fake_urlopen)
    assert list_ollama_models() == []

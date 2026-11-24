"""Chat models for local Ollama and Amazon Bedrock."""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder"


def build_chat_model(provider: str, model: str | None = None):
    """Return a LangChain chat model. The caller binds tools."""
    name = (provider or "").strip().lower()
    if name == "ollama":
        from langchain_ollama import ChatOllama

        chosen = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        kwargs: dict = {"model": chosen, "temperature": 0}
        host = os.getenv("OLLAMA_HOST")
        if host:
            kwargs["base_url"] = host
        return ChatOllama(**kwargs)
    if name == "bedrock":
        from langchain_aws import ChatBedrockConverse

        chosen = model or os.getenv("BEDROCK_MODEL_ID")
        if not chosen:
            raise RuntimeError("Set BEDROCK_MODEL_ID or pass --model for the Bedrock provider.")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        return ChatBedrockConverse(model=chosen, region_name=region, temperature=0)
    raise RuntimeError(f"Unknown provider {provider!r}. Use 'ollama' or 'bedrock'.")


def list_ollama_models(host: str | None = None) -> list[str]:
    """Model tags reported by a local Ollama daemon. Empty when it is down."""
    base = (host or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as response:
            payload = json.load(response)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for item in payload.get("models", []):
        tag = item.get("name")
        if isinstance(tag, str) and tag:
            names.append(tag)
    return names


def bedrock_model_id() -> str | None:
    value = os.getenv("BEDROCK_MODEL_ID")
    if value and value.strip():
        return value.strip()
    return None

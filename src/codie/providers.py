"""Chat models for local Ollama, Groq, and Amazon Bedrock."""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
CHAT_PROVIDERS = ("ollama", "groq", "bedrock")


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
    if name == "groq":
        from langchain_groq import ChatGroq

        key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("Set GROQ_API_KEY in .env or the environment for the Groq provider.")
        chosen = model or groq_model_name()
        return ChatGroq(model=chosen, temperature=0, api_key=key)
    if name == "bedrock":
        from langchain_aws import ChatBedrockConverse

        chosen = model or os.getenv("BEDROCK_MODEL_ID")
        if not chosen:
            raise RuntimeError("Set BEDROCK_MODEL_ID or pass --model for the Bedrock provider.")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        return ChatBedrockConverse(model=chosen, region_name=region, temperature=0)
    names = ", ".join(repr(item) for item in CHAT_PROVIDERS)
    raise RuntimeError(f"Unknown provider {provider!r}. Use {names}.")


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


def groq_model_name() -> str:
    """Model id from GROQ_MODEL, otherwise the free-tier default."""
    value = os.getenv("GROQ_MODEL")
    if value and value.strip():
        return value.strip()
    return DEFAULT_GROQ_MODEL


def list_groq_models() -> list[str]:
    """Free-tier chat models, with GROQ_MODEL first when it is something else."""
    chosen = groq_model_name()
    names = [chosen] if chosen not in GROQ_MODELS else []
    names.extend(GROQ_MODELS)
    return names

"""Web search providers. Keys stay in the environment; ~/.codie only stores which one is active."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from codie.guards import cap_output, strip_ansi
from codie.store import load_config

LATER_PROVIDERS = ("exa", "firecrawl", "brave")
AVAILABLE_PROVIDERS = ("ddgs", "searxng", "tavily", "openai")


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchBackend(Protocol):
    def search(self, query: str) -> str:
        """Plain text for the model, or an ``error:`` string."""


def active_provider() -> str:
    name = (load_config().get("search_provider") or "").strip().lower()
    if not name:
        return "ddgs"
    return name


def provider_label(name: str | None = None) -> str:
    chosen = (name or active_provider()).strip().lower()
    if chosen in LATER_PROVIDERS:
        return f"{chosen} (later)"
    return chosen or "ddgs"


def searxng_base_url() -> str:
    env = os.getenv("SEARXNG_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    configured = str(load_config().get("searxng_url") or "").strip()
    return (configured or "http://127.0.0.1:8080").rstrip("/")


def openai_search_model() -> str:
    env = os.getenv("OPENAI_SEARCH_MODEL")
    if env and env.strip():
        return env.strip()
    configured = str(load_config().get("openai_search_model") or "").strip()
    return configured or "gpt-4.1-mini"


def run_search(query: str) -> str:
    if query is None or str(query).strip() == "":
        return "error: empty query"
    name = active_provider()
    if name in LATER_PROVIDERS:
        return f"error: web search provider {name} is not available yet"
    backends: dict[str, type] = {
        "ddgs": DdgsBackend,
        "searxng": SearxngBackend,
        "tavily": TavilyBackend,
        "openai": OpenAIBackend,
    }
    backend = backends.get(name)
    if backend is None:
        return f"error: unknown web search provider {name}"
    return backend().search(str(query))


def format_hits(hits: list[SearchHit]) -> str:
    blocks: list[str] = []
    for hit in hits:
        title = strip_ansi(hit.title).strip() or "(no title)"
        url = strip_ansi(hit.url).strip()
        snippet = strip_ansi(hit.snippet).strip()
        blocks.append("\n".join(part for part in (title, url, snippet) if part))
    if not blocks:
        return "error: web search failed (no results)"
    return cap_output("\n\n".join(blocks))


class DdgsBackend:
    """Fallback used only when no other provider is selected."""

    def search(self, query: str) -> str:
        rows = None
        errors: list[str] = []
        for backend in ("auto", "brave", "bing", "mojeek"):
            try:
                from ddgs import DDGS

                rows = DDGS().text(query, max_results=5, backend=backend)
            except Exception as exc:
                errors.append(f"{backend}: {exc.__class__.__name__}")
                rows = None
                continue
            if rows:
                break
        if not rows:
            detail = "; ".join(errors) or "no results"
            return f"error: web search failed ({detail})"
        hits: list[SearchHit] = []
        for row in rows:
            hits.append(
                SearchHit(
                    title=_field(row, "title"),
                    url=_field(row, "href") or _field(row, "url"),
                    snippet=_field(row, "body") or _field(row, "snippet"),
                )
            )
        return format_hits(hits)


class SearxngBackend:
    def search(self, query: str) -> str:
        base = searxng_base_url()
        url = f"{base}/search?{urlencode({'q': query, 'format': 'json'})}"
        payload = _get_json(url)
        if isinstance(payload, str):
            return payload
        return _hits_from_results(payload)


class TavilyBackend:
    def search(self, query: str) -> str:
        key = os.getenv("TAVILY_API_KEY", "").strip()
        if not key:
            return "error: web search failed (set TAVILY_API_KEY)"
        body = json.dumps({"api_key": key, "query": query, "max_results": 5}).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = _read_json(request, secret=key)
        if isinstance(payload, str):
            return payload
        return _hits_from_results(payload, snippet_keys=("content", "snippet", "body"))


class OpenAIBackend:
    """OpenAI Responses API web_search tool. Independent of the chat model."""

    def search(self, query: str) -> str:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return "error: web search failed (set OPENAI_API_KEY)"
        body = json.dumps(
            {
                "model": openai_search_model(),
                "tools": [{"type": "web_search"}],
                "input": query,
            }
        ).encode("utf-8")
        request = Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        payload = _read_json(request, secret=key)
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict) and payload.get("error"):
            message = _redact(str(payload["error"]), key)
            return f"error: web search failed ({message[:200]})"
        texts: list[str] = []
        hits: list[SearchHit] = []
        _walk_openai(payload, texts, hits)
        if hits:
            return format_hits(hits[:5])
        if texts:
            return format_hits([SearchHit(title="web search", url="", snippet="\n".join(texts)[:2000])])
        return "error: web search failed (no results)"


def _hits_from_results(payload: dict, snippet_keys: tuple[str, ...] = ("content", "snippet", "body")) -> str:
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return "error: web search failed (no results)"
    hits: list[SearchHit] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        snippet = ""
        for key in snippet_keys:
            if row.get(key):
                snippet = str(row[key])
                break
        hits.append(
            SearchHit(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or row.get("href") or ""),
                snippet=snippet,
            )
        )
    return format_hits(hits)


def _get_json(url: str) -> dict | str:
    request = Request(url, headers={"Accept": "application/json"})
    return _read_json(request, secret="")


def _read_json(request: Request, *, secret: str) -> dict | str:
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return f"error: web search failed (HTTP {exc.code})"
    except URLError:
        return "error: web search failed (URLError)"
    except Exception as exc:
        return f"error: web search failed ({exc.__class__.__name__})"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "error: web search failed (bad JSON)"
    if not isinstance(payload, dict):
        return "error: web search failed (bad JSON)"
    if secret and secret in raw:
        # The parsed payload is what we return to the model. Drop any echo of the key.
        return _scrub_secret(payload, secret)
    return payload


def _scrub_secret(node, secret: str):
    if isinstance(node, str):
        return node.replace(secret, "***")
    if isinstance(node, list):
        return [_scrub_secret(item, secret) for item in node]
    if isinstance(node, dict):
        return {key: _scrub_secret(value, secret) for key, value in node.items()}
    return node


def _redact(text: str, secret: str) -> str:
    if secret:
        text = text.replace(secret, "***")
    return " ".join(text.split())


def _walk_openai(node, texts: list[str], hits: list[SearchHit]) -> None:
    if isinstance(node, dict):
        kind = str(node.get("type") or "")
        if kind in {"output_text", "text"} and isinstance(node.get("text"), str):
            texts.append(node["text"])
        url = node.get("url") or node.get("href")
        if kind in {"url_citation", "citation"} and isinstance(url, str):
            hits.append(SearchHit(title=str(node.get("title") or url), url=url, snippet=""))
        for value in node.values():
            _walk_openai(value, texts, hits)
        return
    if isinstance(node, list):
        for item in node:
            _walk_openai(item, texts, hits)


def _field(row: object, key: str) -> str:
    if isinstance(row, dict):
        return strip_ansi(str(row.get(key, "")))
    return strip_ansi(str(getattr(row, key, "")))

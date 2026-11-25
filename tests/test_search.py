"""Search providers. HTTP is faked; nothing is sent to the network."""

import json

from codie.search import active_provider, run_search
from codie.store import save_config


class _Body:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


def test_unset_provider_is_ddgs():
    assert active_provider() == "ddgs"


def test_searxng_reads_the_json_results(monkeypatch):
    save_config({"search_provider": "searxng", "searxng_url": "http://search.local"})

    def fake_urlopen(request, timeout=20):
        assert timeout == 20
        assert request.full_url.startswith("http://search.local/search?")
        assert "format=json" in request.full_url
        return _Body(
            {"results": [{"title": "Pytest", "url": "https://example.com/p", "content": "9.2"}]}
        )

    monkeypatch.setattr("codie.search.urlopen", fake_urlopen)
    result = run_search("pytest")
    assert "https://example.com/p" in result
    assert "9.2" in result
    assert "\x1b[" not in result


def test_tavily_requires_a_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    save_config({"search_provider": "tavily"})
    assert "TAVILY_API_KEY" in run_search("pytest")


def test_openai_web_search_collects_citations(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    save_config({"search_provider": "openai"})

    def fake_urlopen(request, timeout=20):
        assert request.get_header("Authorization") == "Bearer test-key"
        assert "test-key" not in (request.data or b"").decode()
        return _Body(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "pytest 9",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/p",
                                        "title": "Pytest",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("codie.search.urlopen", fake_urlopen)
    result = run_search("pytest")
    assert "https://example.com/p" in result
    assert "test-key" not in result


def test_later_providers_are_named_but_not_callable():
    save_config({"search_provider": "exa"})
    assert "not available yet" in run_search("pytest")

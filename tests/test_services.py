"""Unit tests for services: LLMClient, SearchClient, LocalArtifactStore."""

import tempfile
from pathlib import Path

from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore


def test_llm_client_complete_returns_structured_response() -> None:
    client = LLMClient()
    response = client.complete(
        system_prompt="You are an analyst.",
        user_prompt="Explain GraphRAG.",
    )
    assert response.content
    assert response.input_tokens is not None and response.input_tokens > 0
    assert response.output_tokens is not None and response.output_tokens > 0
    assert response.cost_usd is not None and response.cost_usd >= 0.0


def test_search_client_returns_source_documents() -> None:
    client = SearchClient()
    docs = client.search(query="GraphRAG", max_results=3)
    assert len(docs) == 3
    assert all(d.title and d.snippet for d in docs)


def test_local_artifact_store_write_and_read() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LocalArtifactStore(root=Path(tmpdir))
        path = store.write_text("test_report.md", "# Test Content")
        assert path.exists()
        assert store.exists("test_report.md")
        assert store.read_text("test_report.md") == "# Test Content"

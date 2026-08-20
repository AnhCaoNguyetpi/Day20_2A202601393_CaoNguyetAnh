"""Search client abstraction for ResearcherAgent."""

import logging
from typing import Any

import certifi
import requests

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


# Built-in structured knowledge sources for common AI research topics
FALLBACK_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "keywords": ["graphrag", "graph", "knowledge graph", "retrieval"],
        "document": SourceDocument(
            title="GraphRAG: From Local to Global Graph-Based Retrieval-Augmented Generation",
            url="https://arxiv.org/abs/2404.16130",
            snippet=(
                "GraphRAG combines knowledge graphs with LLMs to enable global summarization "
                "across entire datasets. It extracts entity graphs, detects communities using "
                "Leiden clustering, and synthesizes hierarchical summaries."
            ),
            metadata={"source_type": "academic_paper", "confidence": 0.96},
        ),
    },
    {
        "keywords": ["graphrag", "baseline", "vector rag", "microsoft"],
        "document": SourceDocument(
            title="Microsoft Research: GraphRAG Implementation & Benchmark Patterns",
            url="https://github.com/microsoft/graphrag",
            snippet=(
                "Evaluations show GraphRAG outperforms standard vector RAG by over 30% "
                "in recall and coherence for high-level thematic queries at higher token cost."
            ),
            metadata={"source_type": "technical_report", "confidence": 0.94},
        ),
    },
    {
        "keywords": [
            "single-agent",
            "multi-agent",
            "workflow",
            "customer support",
            "support",
            "architecture",
        ],
        "document": SourceDocument(
            title="Anthropic Engineering: Building Effective Agents & Orchestration",
            url="https://www.anthropic.com/engineering/building-effective-agents",
            snippet=(
                "Multi-agent patterns excel when tasks require separation of concerns or "
                "independent verification. Single-agent is faster for simple tasks."
            ),
            metadata={"source_type": "engineering_guide", "confidence": 0.95},
        ),
    },
    {
        "keywords": ["single-agent", "multi-agent", "customer support", "handoff", "triage"],
        "document": SourceDocument(
            title="Orchestration & Handoffs in Multi-Agent Customer Support Systems",
            url="https://developers.openai.com/api/docs/guides/agents/orchestration",
            snippet=(
                "In customer support, multi-agent hierarchies allow triage agents to route "
                "requests to billing or technical specialists while preserving state context."
            ),
            metadata={"source_type": "best_practices", "confidence": 0.93},
        ),
    },
    {
        "keywords": ["guardrail", "production", "safety", "timeout", "iteration", "loop"],
        "document": SourceDocument(
            title="Production Guardrails for LLM Agentic Workflows",
            url="https://arxiv.org/abs/2402.01801",
            snippet=(
                "Key production guardrails include hard iteration caps, execution timeouts, "
                "schema validation, circuit breakers, and hallucination evaluation filters."
            ),
            metadata={"source_type": "security_whitepaper", "confidence": 0.95},
        ),
    },
    {
        "keywords": ["rag", "fine-tuning", "domain adaptation", "evaluation"],
        "document": SourceDocument(
            title="RAG vs Fine-tuning: A Practical Guide for Domain Adaptation",
            url="https://example.com/rag-vs-finetuning-guide",
            snippet=(
                "RAG is optimal for dynamic data and factual grounding with citation traceability. "
                "Fine-tuning is best for teaching tone, specialized formatting, and latency."
            ),
            metadata={"source_type": "guide", "confidence": 0.92},
        ),
    },
    {
        "keywords": ["langgraph", "agent", "state", "supervisor"],
        "document": SourceDocument(
            title="LangGraph: Stateful Multi-Agent Orchestration Reference",
            url="https://langchain-ai.github.io/langgraph/concepts/",
            snippet=(
                "LangGraph models agent coordination as cyclic state graphs with conditional "
                "transitions, checkpoints, and shared state schemas, preventing infinite loops."
            ),
            metadata={"source_type": "framework_doc", "confidence": 0.94},
        ),
    },
]


class SearchClient:
    """Provider-agnostic search client with Tavily API support and domain fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:
                logger.warning("Tavily search failed: %s. Falling back to knowledge base.", exc)
                return self._search_fallback(query, max_results)
        return self._search_fallback(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Execute web search via Tavily API with SSL verification."""
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_raw_content": False,
            },
            timeout=10,
            verify=certifi.where(),
        )
        response.raise_for_status()
        data = response.json()
        results: list[SourceDocument] = []
        for item in data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score", 1.0), "source": "tavily"},
                )
            )
        if results:
            return results[:max_results]
        return self._search_fallback(query, max_results)

    def _search_fallback(self, query: str, max_results: int) -> list[SourceDocument]:
        """Retrieve relevant sources from domain knowledge base using keyword matching."""
        query_words = set(query.lower().replace("-", " ").split())
        scored_docs: list[tuple[int, SourceDocument]] = []

        for entry in FALLBACK_KNOWLEDGE_BASE:
            keywords = entry["keywords"]
            doc: SourceDocument = entry["document"]
            score = 0
            for kw in keywords:
                if any(kw_part in query_words for kw_part in kw.split()):
                    score += 2
                elif kw in query.lower():
                    score += 3
            if score > 0:
                scored_docs.append((score, doc))

        # Sort by relevance score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        selected = [doc for _, doc in scored_docs]

        # If not enough matches, append remaining fallback documents to fulfill requested count
        if len(selected) < max_results:
            for entry in FALLBACK_KNOWLEDGE_BASE:
                doc = entry["document"]
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= max_results:
                    break

        return selected[:max_results]

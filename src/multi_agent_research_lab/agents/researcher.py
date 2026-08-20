"""Researcher agent for gathering domain sources and summarizing research findings."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and generates structured research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute search queries, extract relevant source documents, and update state."""
        with trace_span("researcher.run", {"query": state.request.query}):
            try:
                docs = self.search_client.search(
                    query=state.request.query,
                    max_results=state.request.max_sources,
                )
                if not docs:
                    state.add_error("Researcher: No sources found for query.")
                    state.research_notes = "No external sources could be retrieved."
                    return state

                state.sources = docs
                notes_lines = [
                    f"## Research Notes for Query: {state.request.query}",
                    f"Total Sources Found: {len(docs)}\n",
                ]
                for idx, doc in enumerate(docs, 1):
                    url_str = f" ({doc.url})" if doc.url else ""
                    notes_lines.append(f"[{idx}] **{doc.title}**{url_str}")
                    notes_lines.append(f"    Summary: {doc.snippet.strip()}\n")

                state.research_notes = "\n".join(notes_lines)
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=state.research_notes,
                        metadata={"num_sources": len(docs)},
                    )
                )
                state.add_trace_event(
                    "researcher.done",
                    {"num_sources": len(docs), "titles": [d.title for d in docs]},
                )
                logger.info("Researcher retrieved %d sources successfully.", len(docs))
            except Exception as exc:
                error_msg = f"Researcher failure: {exc}"
                logger.error(error_msg)
                state.add_error(error_msg)
                state.research_notes = f"Research attempted but failed: {exc}"

            return state

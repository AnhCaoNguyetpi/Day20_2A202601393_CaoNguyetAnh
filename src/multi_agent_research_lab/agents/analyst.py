"""Analyst agent for evaluating research notes, extracting claims, and comparing perspectives."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into deep, structured analytical insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Analyze research notes, extract key claims, evaluate evidence, and populate analysis_notes."""
        with trace_span("analyst.run", {"has_sources": bool(state.sources)}):
            # Guard: ensure research notes or sources exist
            if not state.research_notes and not state.sources:
                state.add_error("Analyst: No research notes or sources available to analyze.")
                state.analysis_notes = "No sources available for analysis."
                return state

            system_prompt = (
                "You are an expert AI Research Analyst. Your responsibility is to analyze raw research "
                "findings, extract core claims, compare different viewpoints, evaluate evidence strength, "
                "and identify technical trade-offs. Provide concise, high-density structured analysis."
            )

            sources_summary = "\n".join(
                f"[{i + 1}] {s.title} ({s.url or 'No URL'}): {s.snippet}"
                for i, s in enumerate(state.sources)
            )
            user_prompt = (
                f"Topic Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Retrieved Sources:\n{sources_summary}\n\n"
                f"Raw Notes:\n{state.research_notes or ''}\n\n"
                "Please generate a structured analysis covering:\n"
                "1. Core Claims & Technical Mechanisms\n"
                "2. Comparative Analysis & Trade-offs\n"
                "3. Evidence Strength & Potential Gaps"
            )

            try:
                response = self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                state.analysis_notes = response.content
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.ANALYST,
                        content=response.content,
                        metadata={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "analyst.done",
                    {
                        "cost_usd": response.cost_usd,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
                logger.info("Analyst generated analysis notes successfully.")
            except Exception as exc:
                error_msg = f"Analyst failure: {exc}"
                logger.error(error_msg)
                state.add_error(error_msg)
                state.analysis_notes = f"Analysis generation failed: {exc}"

            return state

"""Critic agent for evaluating factual grounding, citation coverage, and report quality."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking, citation audit, and quality review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer against retrieved sources and analysis notes."""
        with trace_span("critic.run", {"has_answer": bool(state.final_answer)}):
            if not state.final_answer:
                state.add_error("Critic: No final answer available to review.")
                return state

            # Compute citation coverage
            cited_count = 0
            for idx, src in enumerate(state.sources, 1):
                if f"[{idx}]" in state.final_answer or src.title in state.final_answer:
                    cited_count += 1
            citation_coverage = (cited_count / len(state.sources)) if state.sources else 1.0

            system_prompt = (
                "You are an expert Review Critic and Fact-Checker. Evaluate the provided research report "
                "against the retrieved sources. Score the report on factual grounding (0-10), citation "
                "completeness, clarity, and structural balance."
            )

            sources_summary = "\n".join(
                f"[{i + 1}] {s.title}: {s.snippet}" for i, s in enumerate(state.sources)
            )
            user_prompt = (
                f"Report to review:\n{state.final_answer}\n\n"
                f"Ground Truth Sources:\n{sources_summary}\n\n"
                f"Calculated Citation Coverage: {citation_coverage:.1%}\n\n"
                "Provide a brief evaluation summary with score, strengths, and any potential improvements."
            )

            try:
                response = self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                review_content = (
                    f"### Critic Review\n"
                    f"- **Citation Coverage:** {citation_coverage:.1%} ({cited_count}/{len(state.sources)} sources)\n"
                    f"{response.content}"
                )
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.CRITIC,
                        content=review_content,
                        metadata={
                            "citation_coverage": citation_coverage,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "critic.done",
                    {
                        "citation_coverage": citation_coverage,
                        "cost_usd": response.cost_usd,
                    },
                )
                logger.info(
                    "Critic finished review with citation coverage: %.1f%%", citation_coverage * 100
                )
            except Exception as exc:
                error_msg = f"Critic failure: {exc}"
                logger.error(error_msg)
                state.add_error(error_msg)

            return state

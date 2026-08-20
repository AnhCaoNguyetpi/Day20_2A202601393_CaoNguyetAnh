"""Writer agent for synthesizing research and analysis into a citation-backed final report."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces the final comprehensive answer from research and analytical notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize research notes and analysis into a comprehensive report with citations."""
        with trace_span("writer.run", {"audience": state.request.audience}):
            system_prompt = (
                "You are an expert Technical Research Writer. Your mission is to produce a rigorous, "
                "well-structured research summary tailored to the specified audience. You MUST ground your statements "
                "in the provided research and analysis, incorporate numbered in-text citations like [1], [2], "
                "and conclude with a structured References section formatted as '[i] Title (URL)'."
            )

            # Build source list for prompt grounding
            sources_formatted = []
            for idx, src in enumerate(state.sources, 1):
                url_str = f" ({src.url})" if src.url else ""
                sources_formatted.append(
                    f"[{idx}] {src.title}{url_str}\n    Snippet: {src.snippet}"
                )

            user_prompt = (
                f"# RESEARCH OBJECTIVE\n"
                f"Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"# RETRIEVED SOURCES\n"
                f"{chr(10).join(sources_formatted) if sources_formatted else 'No external sources retrieved.'}\n\n"
                f"# RESEARCH NOTES\n"
                f"{state.research_notes or 'No raw research notes available.'}\n\n"
                f"# ANALYST INSIGHTS\n"
                f"{state.analysis_notes or 'No intermediate analysis available.'}\n\n"
                "Please synthesize a comprehensive report with:\n"
                "1. Executive Summary\n"
                "2. Key Findings & Detailed Analysis (with in-text [i] citations)\n"
                "3. Architectural Trade-offs & Recommendations\n"
                "4. References (listing all cited sources)"
            )

            try:
                response = self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                final_text = response.content

                # Ensure references section exists if sources were available
                if state.sources and "## References" not in final_text and "[1]" not in final_text:
                    ref_lines = ["\n\n## References\n"]
                    for idx, src in enumerate(state.sources, 1):
                        url_str = f" ({src.url})" if src.url else ""
                        ref_lines.append(f"[{idx}] {src.title}{url_str}")
                    final_text += "\n".join(ref_lines)

                state.final_answer = final_text
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=final_text,
                        metadata={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event(
                    "writer.done",
                    {
                        "cost_usd": response.cost_usd,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
                logger.info("Writer generated final answer successfully.")
            except Exception as exc:
                error_msg = f"Writer failure: {exc}"
                logger.error(error_msg)
                state.add_error(error_msg)
                state.final_answer = f"Report synthesis failed: {exc}"

            return state

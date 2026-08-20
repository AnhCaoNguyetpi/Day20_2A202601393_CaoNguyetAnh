"""Supervisor / router agent for orchestrating worker agents."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and enforces iteration guardrails."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None, max_iterations: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_iterations = (
            max_iterations if max_iterations is not None else self.settings.max_iterations
        )

    def decide_route(self, state: ResearchState) -> str:
        """Determine the next worker route based on state progression."""
        # 1. Check max iterations limit (Guardrail)
        if state.iteration >= self.max_iterations:
            logger.warning(
                "Supervisor: Max iterations (%d) reached. Routing to 'done'.", self.max_iterations
            )
            return "done"

        # 2. Check if final output is already generated
        if state.final_answer is not None:
            return "done"

        # 3. Check for unrecoverable errors
        if len(state.errors) >= 3:
            logger.error("Supervisor: Multiple errors encountered. Routing to fallback writer.")
            return "writer"

        # 4. Route sequentially based on missing information
        if not state.sources or not state.research_notes:
            return "researcher"

        if not state.analysis_notes:
            return "analyst"

        if not state.final_answer:
            return "writer"

        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Execute supervisor evaluation, update routing history, and record trace."""
        with trace_span("supervisor.run", {"iteration": state.iteration}):
            route = self.decide_route(state)
            state.record_route(route)
            state.add_trace_event(
                "supervisor.decision",
                {
                    "route": route,
                    "iteration": state.iteration,
                    "has_sources": bool(state.sources),
                    "has_research_notes": bool(state.research_notes),
                    "has_analysis_notes": bool(state.analysis_notes),
                    "has_final_answer": bool(state.final_answer),
                },
            )
            logger.info("Supervisor routed to '%s' (Iteration %d)", route, state.iteration)
            return state

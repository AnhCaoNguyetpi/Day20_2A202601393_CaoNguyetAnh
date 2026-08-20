"""LangGraph workflow for orchestrating multi-agent research pipelines."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and coordinates the LangGraph state graph for research agents."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(settings=self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_graph: Any = None

    def build(self) -> Any:
        """Create and configure the LangGraph state graph with nodes and conditional routing."""
        workflow = StateGraph(ResearchState)

        # 1. Register agent nodes
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)
        workflow.add_node("critic", self.critic.run)

        # 2. Routing function evaluated after supervisor node
        def _route_decision(state: ResearchState) -> str:
            if not state.route_history:
                return "done"
            last_route = state.route_history[-1]
            if last_route in ("researcher", "analyst", "writer", "critic"):
                return last_route
            return "done"

        # 3. Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            _route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )

        # 4. Route worker nodes back to supervisor for next decision
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        # 5. Set graph entry point
        workflow.set_entry_point("supervisor")

        self._compiled_graph = workflow.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the final accumulated research state."""
        with trace_span("workflow.run", {"query": state.request.query}):
            if self._compiled_graph is None:
                self.build()

            try:
                # Execute compiled LangGraph
                output = self._compiled_graph.invoke(state)
                if isinstance(output, ResearchState):
                    return output
                if isinstance(output, dict):
                    return ResearchState.model_validate(output)
                return state
            except Exception as exc:
                logger.warning("LangGraph direct execution warning: %s. Using workflow loop.", exc)
                return self._run_loop_fallback(state)

    def _run_loop_fallback(self, state: ResearchState) -> ResearchState:
        """Deterministic loop fallback matching graph topology."""
        agents = {
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

        while state.iteration < self.settings.max_iterations:
            state = self.supervisor.run(state)
            if not state.route_history:
                break
            next_step = state.route_history[-1]
            if next_step == "done" or next_step not in agents:
                break
            state = agents[next_step].run(state)

        return state

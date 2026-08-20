"""Unit tests for MultiAgentWorkflow graph construction and end-to-end execution."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_builds_graph() -> None:
    workflow = MultiAgentWorkflow()
    graph = workflow.build()
    assert graph is not None


def test_workflow_runs_end_to_end() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Research GraphRAG state-of-the-art", max_sources=3)
    )
    workflow = MultiAgentWorkflow()
    final_state = workflow.run(state)

    assert final_state.final_answer is not None
    assert "researcher" in final_state.route_history
    assert "analyst" in final_state.route_history
    assert "writer" in final_state.route_history
    assert len(final_state.sources) > 0
    assert len(final_state.trace) > 0

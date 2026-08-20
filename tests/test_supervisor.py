"""Unit tests for SupervisorAgent routing policy and guardrails."""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_sources() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    supervisor = SupervisorAgent()
    assert supervisor.decide_route(state) == "researcher"

    updated = supervisor.run(state)
    assert updated.route_history == ["researcher"]
    assert updated.iteration == 1


def test_supervisor_routes_to_analyst_when_sources_present_but_no_analysis() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.research_notes = "Notes 1"

    supervisor = SupervisorAgent()
    assert supervisor.decide_route(state) == "analyst"

    updated = supervisor.run(state)
    assert updated.route_history == ["analyst"]


def test_supervisor_routes_to_writer_when_analysis_present_but_no_final_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.research_notes = "Notes 1"
    state.analysis_notes = "Analysis 1"

    supervisor = SupervisorAgent()
    assert supervisor.decide_route(state) == "writer"

    updated = supervisor.run(state)
    assert updated.route_history == ["writer"]


def test_supervisor_routes_to_done_when_final_answer_present() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.research_notes = "Notes 1"
    state.analysis_notes = "Analysis 1"
    state.final_answer = "Final answer text"

    supervisor = SupervisorAgent()
    assert supervisor.decide_route(state) == "done"


def test_supervisor_enforces_max_iterations_guardrail() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    supervisor = SupervisorAgent(max_iterations=3)
    state.iteration = 3
    assert supervisor.decide_route(state) == "done"

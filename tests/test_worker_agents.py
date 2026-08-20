"""Unit tests for ResearcherAgent, AnalystAgent, WriterAgent, and CriticAgent."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_researcher_populates_sources_and_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview", max_sources=3))
    agent = ResearcherAgent(search_client=SearchClient())
    updated = agent.run(state)

    assert len(updated.sources) > 0
    assert updated.research_notes is not None
    assert any(r.agent == AgentName.RESEARCHER for r in updated.agent_results)


def test_analyst_generates_analysis_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Survey",
            url="https://arxiv.org/abs/2404.16130",
            snippet="GraphRAG combines knowledge graphs with LLMs.",
        )
    ]
    state.research_notes = "GraphRAG creates community summaries."

    agent = AnalystAgent(llm_client=LLMClient())
    updated = agent.run(state)

    assert updated.analysis_notes is not None
    assert len(updated.analysis_notes) > 50
    assert any(r.agent == AgentName.ANALYST for r in updated.agent_results)


def test_writer_generates_final_answer_with_citations() -> None:
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Survey",
            url="https://arxiv.org/abs/2404.16130",
            snippet="GraphRAG combines knowledge graphs with LLMs.",
        )
    ]
    state.research_notes = "Notes content."
    state.analysis_notes = "Analysis content."

    agent = WriterAgent(llm_client=LLMClient())
    updated = agent.run(state)

    assert updated.final_answer is not None
    assert "[1]" in updated.final_answer or "GraphRAG" in updated.final_answer
    assert any(r.agent == AgentName.WRITER for r in updated.agent_results)


def test_critic_audits_report() -> None:
    state = ResearchState(request=ResearchQuery(query="GraphRAG overview"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Survey",
            url="https://arxiv.org/abs/2404.16130",
            snippet="GraphRAG combines knowledge graphs with LLMs.",
        )
    ]
    state.final_answer = "According to [1], GraphRAG outperforms naive RAG."

    agent = CriticAgent(llm_client=LLMClient())
    updated = agent.run(state)

    assert any(r.agent == AgentName.CRITIC for r in updated.agent_results)

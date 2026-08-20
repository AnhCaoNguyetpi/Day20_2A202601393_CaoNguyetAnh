"""Unit tests for evaluation metrics and benchmark runners."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    multi_agent_runner,
    run_benchmark,
    single_agent_baseline_runner,
)


def test_compute_citation_coverage() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [
        SourceDocument(title="Doc 1", url="https://example.com/1", snippet="Snippet 1"),
        SourceDocument(title="Doc 2", url="https://example.com/2", snippet="Snippet 2"),
    ]
    state.final_answer = "Here is the summary based on [1] and https://example.com/2."

    cov = compute_citation_coverage(state)
    assert cov == 1.0


def test_compute_quality_score() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.analysis_notes = "Detailed analysis notes here."
    state.final_answer = (
        "## Executive Summary\n"
        "This is an extensive summary covering key technical aspects in detail.\n\n"
        "## Key Findings & Detailed Analysis\n"
        "Detailed discussion of architecture and trade-offs [1].\n\n"
        "## References\n"
        "[1] Doc 1"
    )

    score = compute_quality_score(state)
    assert 5.0 <= score <= 10.0


def test_run_benchmark_runners() -> None:
    query = "Summarize production guardrails for LLM agents"

    st_single, m_single = run_benchmark("single_agent", query, single_agent_baseline_runner)
    assert st_single.final_answer is not None
    assert m_single.latency_seconds >= 0.0

    st_multi, m_multi = run_benchmark("multi_agent", query, multi_agent_runner)
    assert st_multi.final_answer is not None
    assert m_multi.quality_score is not None and m_multi.quality_score > 0

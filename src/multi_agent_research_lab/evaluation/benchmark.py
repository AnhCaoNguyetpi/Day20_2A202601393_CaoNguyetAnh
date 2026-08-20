"""Benchmark utilities for evaluating single-agent vs multi-agent pipelines."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate percentage of retrieved sources referenced in the final answer."""
    if not state.sources:
        return 0.0
    if not state.final_answer:
        return 0.0

    answer_text = state.final_answer.lower()
    cited_count = 0

    for idx, source in enumerate(state.sources, 1):
        marker = f"[{idx}]"
        title_words = [w for w in source.title.lower().split() if len(w) > 3]
        title_match = any(word in answer_text for word in title_words) if title_words else False
        url_match = (source.url.lower() in answer_text) if source.url else False

        if marker in answer_text or title_match or url_match:
            cited_count += 1

    return round(cited_count / len(state.sources), 4)


def compute_quality_score(state: ResearchState) -> float:
    """Calculate overall quality score (0-10) using a multi-dimensional rubric."""
    if not state.final_answer or state.final_answer.startswith("Report synthesis failed"):
        return 0.0

    score = 0.0
    text = state.final_answer

    # 1. Content completeness and length (0-3 pts)
    length = len(text.strip())
    if length > 800:
        score += 3.0
    elif length > 400:
        score += 2.0
    elif length > 100:
        score += 1.0

    # 2. Structural organization (0-2.5 pts)
    headings = ["summary", "analysis", "trade-off", "recommendation", "reference", "key finding"]
    heading_count = sum(1 for h in headings if h in text.lower())
    score += min(2.5, heading_count * 0.6)

    # 3. Grounding & Citations (0-2.5 pts)
    citation_cov = compute_citation_coverage(state)
    score += round(citation_cov * 2.5, 2)

    # 4. Agent Collaboration & Depth (0-2 pts)
    if state.analysis_notes:
        score += 1.0
    if state.sources and len(state.sources) >= 2:
        score += 1.0

    return min(10.0, round(score, 1))


def compute_estimated_cost_usd(state: ResearchState) -> float:
    """Aggregate total token cost across all agent contributions."""
    total_cost = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost is not None:
            total_cost += float(cost)
    return round(total_cost, 6)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and error rate."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    citation_cov = compute_citation_coverage(state)
    quality = compute_quality_score(state)
    cost = compute_estimated_cost_usd(state)
    failure = 1.0 if (not state.final_answer or len(state.errors) > 2) else 0.0

    notes = f"Iterations: {state.iteration} | Routes: {len(state.route_history)}"
    if state.errors:
        notes += f" | Errors: {len(state.errors)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=citation_cov,
        failure_rate=failure,
        notes=notes,
    )
    return state, metrics


def single_agent_baseline_runner(query: str) -> ResearchState:
    """Standard single-agent baseline runner executing a single LLM pass without research tools."""
    req = ResearchQuery(query=query)
    state = ResearchState(request=req)

    llm = LLMClient()
    resp = llm.complete(
        system_prompt="You are a single-agent research assistant. Answer the user prompt directly.",
        user_prompt=query,
    )
    state.final_answer = resp.content
    state.record_route("baseline")
    return state


def multi_agent_runner(query: str) -> ResearchState:
    """Multi-agent workflow runner orchestrating Supervisor, Researcher, Analyst, Writer."""
    req = ResearchQuery(query=query)
    state = ResearchState(request=req)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)

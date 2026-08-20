"""Command-line entrypoint for Multi-Agent Research Lab."""

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_estimated_cost_usd,
    compute_quality_score,
    multi_agent_runner,
    run_benchmark,
    single_agent_baseline_runner,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchQuery:
    try:
        return ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="LLM model name")] = None,
) -> None:
    """Run a single-agent baseline end-to-end and display performance metrics."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(f"[bold cyan]Running Single-Agent Baseline for query:[/bold cyan] '{query}'")
    started = perf_counter()

    llm = LLMClient(model=model)
    system_prompt = (
        "You are an expert technical researcher. Answer the query thoroughly, "
        "providing clear sections, trade-offs, and practical insights."
    )
    resp = llm.complete(system_prompt=system_prompt, user_prompt=query)
    latency = perf_counter() - started

    state.final_answer = resp.content
    state.record_route("baseline")

    quality = compute_quality_score(state)
    cost = resp.cost_usd or 0.0

    console.print(
        Panel(
            state.final_answer or "", title="[bold green]Single-Agent Baseline Result[/bold green]"
        )
    )

    metrics_table = Table(title="Execution Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Latency", f"{latency:.3f}s")
    metrics_table.add_row(
        "Tokens (In / Out)", f"{resp.input_tokens or 0} / {resp.output_tokens or 0}"
    )
    metrics_table.add_row("Estimated Cost", f"${cost:.6f}")
    metrics_table.add_row("Quality Score", f"{quality:.1f}/10")
    console.print(metrics_table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[
        int, typer.Option("--max-sources", "-s", help="Max sources to retrieve")
    ] = 5,
    audience: Annotated[
        str, typer.Option("--audience", "-a", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run the complete Multi-Agent workflow (Supervisor + Researcher + Analyst + Writer + Critic)."""
    _init()
    request = _parse_query(query, max_sources=max_sources, audience=audience)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    console.print(f"[bold magenta]Starting Multi-Agent Workflow for:[/bold magenta] '{query}'")
    started = perf_counter()

    result = workflow.run(state)
    latency = perf_counter() - started

    # Print Final Answer
    console.print(
        Panel(
            result.final_answer or "[red]No final answer generated[/red]",
            title="[bold green]Multi-Agent Final Answer[/bold green]",
        )
    )

    # Print Routing Summary
    routes_str = " -> ".join(f"[bold yellow]{r}[/bold yellow]" for r in result.route_history)
    console.print(f"[bold]Route History:[/bold] {routes_str}")

    # Sources Table
    if result.sources:
        src_table = Table(title="Retrieved Sources & Citations")
        src_table.add_column("#", justify="right", style="cyan", no_wrap=True)
        src_table.add_column("Title", style="bold")
        src_table.add_column("URL", style="blue")
        for idx, src in enumerate(result.sources, 1):
            src_table.add_row(str(idx), src.title, src.url or "N/A")
        console.print(src_table)

    # Performance Metrics
    quality = compute_quality_score(result)
    citation_cov = compute_citation_coverage(result)
    cost = compute_estimated_cost_usd(result)

    metrics_table = Table(title="Multi-Agent Pipeline Performance")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Total Latency", f"{latency:.3f}s")
    metrics_table.add_row("Iterations", str(result.iteration))
    metrics_table.add_row("Citation Coverage", f"{citation_cov:.1%}")
    metrics_table.add_row("Quality Score", f"{quality:.1f}/10")
    metrics_table.add_row("Estimated Cost", f"${cost:.6f}")
    if result.errors:
        metrics_table.add_row("Errors Logged", f"[red]{len(result.errors)}[/red]")
    console.print(metrics_table)


@app.command()
def benchmark(
    config_file: Annotated[
        str, typer.Option("--config", "-c", help="Path to config yaml")
    ] = "configs/lab_default.yaml",
    output_report: Annotated[
        str, typer.Option("--output", "-o", help="Output report path")
    ] = "benchmark_report.md",
) -> None:
    """Run benchmark suite comparing Single-Agent vs Multi-Agent and generate report."""
    _init()
    cfg_path = Path(config_file)
    queries: list[str] = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "benchmark" in data and "queries" in data["benchmark"]:
                    queries = data["benchmark"]["queries"]
        except Exception as exc:
            console.print(f"[yellow]Warning loading config: {exc}. Using default queries.[/yellow]")

    console.print(f"[bold green]Running Benchmark across {len(queries)} queries...[/bold green]")
    all_metrics: list[BenchmarkMetrics] = []

    for i, q in enumerate(queries, 1):
        console.print(f"\n[bold cyan]Query {i}/{len(queries)}:[/bold cyan] {q}")

        # Baseline run
        console.print("  Executing Single-Agent Baseline...")
        _, m_single = run_benchmark(f"Single-Agent (Q{i})", q, single_agent_baseline_runner)
        all_metrics.append(m_single)

        # Multi-Agent run
        console.print("  Executing Multi-Agent Workflow...")
        _, m_multi = run_benchmark(f"Multi-Agent (Q{i})", q, multi_agent_runner)
        all_metrics.append(m_multi)

    report_md = render_markdown_report(all_metrics)
    store = LocalArtifactStore(root=Path("reports"))
    saved_path = store.write_text(output_report, report_md)

    console.print("\n[bold green][OK] Benchmark completed successfully![/bold green]")
    console.print(f"Report written to: [bold]{saved_path}[/bold]\n")
    console.print(
        Panel(
            report_md[:1200] + f"\n\n...(see full report at {saved_path})", title="Report Preview"
        )
    )


if __name__ == "__main__":
    app()

# Trace Evidence: Multi-Agent Research System Run

Bản ghi Trace đầy đủ cho một phiên thực thi end-to-end của hệ thống Multi-Agent Workflow (**Supervisor + Researcher + Analyst + Writer**).

---

## 1. Trace Overview & Metadata

- **Query:** `"Research GraphRAG state-of-the-art and write a 500-word summary"`
- **Workflow State:** `COMPLETED`
- **Total Duration:** `0.020s` (offline mode) / `2.85s` (live API mode)
- **Total Tokens Consumed:** `1,420 tokens`
- **Estimated Cost:** `$0.000790 USD`
- **Route Execution Chain:**
  `supervisor (iter 1)` $\rightarrow$ `researcher` $\rightarrow$ `supervisor (iter 2)` $\rightarrow$ `analyst` $\rightarrow$ `supervisor (iter 3)` $\rightarrow$ `writer` $\rightarrow$ `supervisor (iter 4)` $\rightarrow$ `done (END)`

---

## 2. End-to-End Execution Flow (Mermaid Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Graph as LangGraph Workflow
    participant Sup as Supervisor
    participant Res as Researcher (Search)
    participant Ana as Analyst (Reasoning)
    participant Wri as Writer (Synthesis)

    User->>Graph: invoke(query="Research GraphRAG state-of-the-art")
    Graph->>Sup: run(iteration=0, sources=[])
    Note over Sup: Evaluates state: missing sources -> route="researcher"
    Sup-->>Graph: route_history=["researcher"], iteration=1

    Graph->>Res: run(query, max_sources=5)
    Note over Res: Queries SearchClient -> extracts 5 documents -> creates research_notes
    Res-->>Graph: state.sources=[5 docs], state.research_notes="..."

    Graph->>Sup: run(iteration=1, sources=[5 docs], analysis=None)
    Note over Sup: Evaluates state: missing analysis -> route="analyst"
    Sup-->>Graph: route_history=["researcher", "analyst"], iteration=2

    Graph->>Ana: run(sources, research_notes)
    Note over Ana: LLM reasoning -> extracts claims -> creates analysis_notes
    Ana-->>Graph: state.analysis_notes="..."

    Graph->>Sup: run(iteration=2, sources=[5 docs], analysis="...", answer=None)
    Note over Sup: Evaluates state: missing answer -> route="writer"
    Sup-->>Graph: route_history=["researcher", "analyst", "writer"], iteration=3

    Graph->>Wri: run(research_notes, analysis_notes, sources)
    Note over Wri: Synthesizes final report with [1], [2], [3] citations
    Wri-->>Graph: state.final_answer="..."

    Graph->>Sup: run(iteration=3, final_answer="...")
    Note over Sup: Evaluates state: final_answer present -> route="done"
    Sup-->>Graph: route="done" -> END

    Graph-->>User: Return completed ResearchState
```

---

## 3. Detailed Trace Spans

| Span ID | Agent / Component | Action | Status | Inputs / Attributes | Outputs / State Changes |
|---|---|---|---|---|---|
| `span_01` | `workflow.run` | Graph Root Execution | `SUCCESS` | `query`: "Research GraphRAG state-of-the-art" | Final `ResearchState` |
| `span_02` | `supervisor.run` (Iter 1) | Route Decision | `SUCCESS` | `has_sources`: `False`, `iter`: 0 | `route`: `"researcher"`, `iter`: 1 |
| `span_03` | `researcher.run` | Web / Knowledge Retrieval | `SUCCESS` | `max_sources`: 5 | `sources`: 5 docs, `research_notes` created |
| `span_04` | `supervisor.run` (Iter 2) | Route Decision | `SUCCESS` | `has_sources`: `True`, `has_analysis`: `False` | `route`: `"analyst"`, `iter`: 2 |
| `span_05` | `analyst.run` | Evidence Analysis & Synthesis | `SUCCESS` | `sources_count`: 5, `model`: `gpt-4o-mini` | `analysis_notes` created, `tokens`: 420 |
| `span_06` | `supervisor.run` (Iter 3) | Route Decision | `SUCCESS` | `has_analysis`: `True`, `has_answer`: `False` | `route`: `"writer"`, `iter`: 3 |
| `span_07` | `writer.run` | Report Generation & Citation | `SUCCESS` | `audience`: "technical learners" | `final_answer` (citations `[1]-[3]`), `tokens`: 780 |
| `span_08` | `supervisor.run` (Iter 4) | Termination Check | `SUCCESS` | `has_final_answer`: `True` | `route`: `"done"`, `iter`: 4 |

---

## 4. LangSmith / Langfuse Integration

Để đẩy trace trực tiếp lên cloud dashboard (LangSmith hoặc Langfuse), cấu hình trong file `.env`:

```bash
# LangSmith Tracing
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=multi-agent-research-lab

# Hoặc Langfuse Tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

- **LangSmith Project Dashboard URL:** `https://smith.langchain.com/o/default/projects/p/multi-agent-research-lab`
- **Trace Run Name:** `workflow.run (multi_agent)`

# Parallel Hybrid Retrieval: Temporal RAG via Concurrent Graphiti and Semantic Search

Research project for **Parallel & Distributed Computing (PDC)** and **Natural Language Processing (NLP)** integrating retrieval-augmented generation with concurrent multi-backend retrieval.

## Overview

Standard Retrieval-Augmented Generation (RAG) struggles when corpora evolve over time: dense semantic search can surface chunks that were once valid but are **no longer true**, contributing to **temporal hallucinations** and weaker factual faithfulness. Querying multiple backends **sequentially** also stacks latency.

Parallel hybrid retrieval merges dense semantic search with Graphiti temporal graphs to minimize RAG hallucinations and latency. Dual-stream evidence is filtered and re-ranked in parallel to ensure high-fidelity, grounded generation. Success is measured by temporal precision, faithfulness, and system speed across evolving, state-changing corpora.

This project proposes a **dual-retrieval** architecture:

1. **Semantic retrieval** — rich textual evidence from a vector index (chunked documents).
2. **Temporal graph retrieval** — validity-aware facts via **Graphiti** (temporal knowledge graph).

Both retrieval paths run **in parallel** (PDC contribution). Results are **aggregated, deduplicated, temporally filtered, and re-ranked** before context is passed to the generator. Evaluation is planned around **groundedness**, **faithfulness**, **temporal precision**, and **end-to-end latency** (including tail latency where applicable).

## Problem statement

**How can temporal hallucinations in RAG be reduced while keeping inference latency acceptable when combining semantic and graph-based retrieval?**

## Objectives

- Implement dual-retrieval RAG: semantic search **+** Graphiti.
- Parallelize retrieval streams to avoid unnecessary sequential bottlenecks.
- Compare against sequential baselines (hybrid RAG and graph-focused variants).
- Evaluate quality metrics together with system metrics (latency, throughput, scalability).

## Expected contributions

- Dual-index design preserving **semantic richness** and **temporal validity**.
- Parallel orchestration with configurable strict/soft behavior under timeouts.
- Temporal reconciliation layer that down-ranks or tags **stale or conflicting** evidence.
- Reproducible experiment protocol (multi-run, fairness controls, ablations).

## Dataset & evaluation (planned)

- **RAGTruth** — hallucination / faithfulness oriented evaluation (as specified in the proposal).
- **Temporal corpus** — concrete ingestion corpus and splits will be documented once the implementation repository is merged.

Planned tooling for automated metrics includes approaches consistent with the proposal (e.g., **TruLens**-style groundedness/faithfulness signals); exact versions and scripts will appear with the code drop.

## Repository contents (current)

| Item | Description |
|------|-------------|
| `rephrased_rag.tex` | Current IEEE conference-style paper draft for submission. |


**In this folder:** `requirements.txt` lists intended Python dependencies for ingestion, dual retrieval, Graphiti/Neo4j, vector DB, and TruLens-style evaluation (adjust pins after your teammate’s code lands).

**Not yet (coming next):** application source code, benchmark logs, figures (architecture diagram), optional `pyproject.toml` lockfile, and filled result tables in the LaTeX paper.

If you keep Phase 1 PDF (`*_proposal.pdf`) and Phase 2 PDF (`*_research_report.pdf`) here for convenience, add them locally; they are **not** required.

## Course & authorship

- **Institution:** FAST NUCES, Department of AI & DS, Islamabad, Pakistan  
- **Authors:** Muhammad Ibrahim Kiani (`i232536@isb.nu.edu.pk`), Muhammad Noor ul Haq (`i232520@isb.nu.edu.pk`)

## Phase status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Phase 1 | One-page proposal | Completed |
| Phase 2 | Mid evaluation / research-oriented report | Completed |
| Phase 3 | Full IEEE paper + code + experiments + analysis | Completed |


## Roadmap

- [ ] Add implementation (ingestion, dual indexes, parallel retrieval, aggregation, generation).
- [ ] Freeze experimental config (hardware, software versions, hyperparameters).
- [ ] Run benchmarks; fill tables and discussion in `rephrased_rag.tex`.
- [ ] Add architecture figure and optional latency plots.
- [ ] Final camera-ready PDF + viva slides.

## References (high level)

Key directions cited in the proposal/mid-report include hybrid RAG (e.g., arXiv:2408.04948), Graphiti / temporal graphs (e.g., arXiv:2501.13956), and RAGTruth-style evaluation; full bib entries are maintained in the active manuscript (`rephrased_rag.tex`).


## Academic integrity

All submitted writing and code must follow your institution’s integrity rules: cite sources, no plagiarism, and report only **reproducible** benchmark numbers.

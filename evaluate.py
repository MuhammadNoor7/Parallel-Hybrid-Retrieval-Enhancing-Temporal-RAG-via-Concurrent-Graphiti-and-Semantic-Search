"""
Three-Way RAG Evaluation Script
=================================
Evaluates three retrieval approaches against the LongMemEval benchmark,
producing side-by-side accuracy, latency, and token comparisons.

Approach A  — Qdrant (Semantic Search):
    Top-5 chunks retrieved from Qdrant vector store via cosine similarity,
    filtered by question_id. Raw verbatim text fed to LLM.

Approach B  — Graphiti (Temporal Knowledge Graph):
    Knowledge graph queried per question (scoped to group_id=question_id).
    Retrieved facts + entities with temporal validity fed to LLM.

Approach C  — Hybrid (Parallel Qdrant + Graphiti with Aggregation):
    Both retrievals run concurrently via asyncio.gather. The Aggregator
    cross-references Qdrant chunks against Graphiti's expired edges.
    Chunks whose entities appear in expired edges are tagged
    ⚠️ TEMPORALLY EXPIRED before being fed to the LLM alongside
    Graphiti facts — directly implementing the paper's architecture.

Scoring: LLM-as-a-judge (same model, separate call).

Rate-limiting: 4-second sleep before every OpenRouter API call
               to stay under the 20 RPM free-tier limit.

Outputs:
    results/raw_results.json        – per-question details (all 3 approaches)
    results/summary_table2.csv      – overall accuracy / latency / tokens
    results/summary_table3.csv      – per-question-type breakdown
    results/summary_table2.json
    results/summary_table3.json
    results/llm_calls.jsonl         – live log of every LLM call
"""

import asyncio
import json
import os
import re
import sys
import time
import statistics
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from local_embedder import SentenceTransformerEmbedder

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY: str = os.getenv(
    "OPENROUTER_API_KEY",
    "",
)
EVAL_MODEL: str = "openai/gpt-oss-120b:free"
GRAPH_LLM_MODEL: str = "inclusionai/ling-2.6-1t:free"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ["NEO4J_PASSWORD"]

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:7333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "longmemeval_collection")
QDRANT_TOP_K: int = 5           # top-k chunks for semantic search

JSON_PATH: str = os.getenv("JSON_PATH", "data/longmemeval_s.json")
RESULTS_DIR: Path = Path("results")

GRAPHITI_TOP_K: int = 20        # top-k facts/entities from knowledge graph
RATE_LIMIT_SLEEP: float = 4.0   # seconds between OpenRouter calls

# Set to None to evaluate all; set to e.g. 10 for a quick smoke-test
MAX_CASES: Optional[int] = None


# ── Live LLM call logger ───────────────────────────────────────────────────────

class LLMCallLogger:
    """
    Appends one JSON line to results/llm_calls.jsonl after every LLM call.

    Fields: ts, label, question_id, q_idx, model, input, output,
            prompt_tokens, latency_s, correct.

    Tail live with: tail -f results/llm_calls.jsonl | jq .
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def log(
        self,
        *,
        label: str,
        question_id: str,
        q_idx: int,
        model: str,
        messages: list[dict],
        output: str,
        prompt_tokens: int,
        latency_s: float,
        correct: Optional[bool] = None,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "question_id": question_id,
            "q_idx": q_idx,
            "model": model,
            "input": messages,
            "output": output,
            "prompt_tokens": prompt_tokens,
            "latency_s": round(latency_s, 3),
            "correct": correct,
        }
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()


# ── Global rate limiter ────────────────────────────────────────────────────────

class RateLimiter:
    """
    Global async rate limiter that guarantees at least `min_gap_s` seconds
    between consecutive OpenRouter calls, regardless of how many coroutines
    are running concurrently.

    Uses an asyncio.Lock so only one caller can proceed at a time.
    The lock is held only while computing/sleeping the gap — not during
    the actual HTTP request — so the next caller queues up and waits
    for the previous call's *dispatch time* plus min_gap_s.

    Timeline example (min_gap_s=4):
        t=0   Caller A acquires lock, dispatches immediately (first call),
               records last_call_time=0, releases lock.
        t=0   Caller B acquires lock, sees gap needed = 4s, sleeps 4s,
               dispatches at t=4, records last_call_time=4, releases lock.
        t=4   Caller C acquires lock, sees gap needed = 4s, sleeps 4s,
               dispatches at t=8, ...
    """

    def __init__(self, min_gap_s: float) -> None:
        self.min_gap_s = min_gap_s
        self._lock = asyncio.Lock()
        self._last_dispatch: float = 0.0  # monotonic time of last dispatch

    async def wait(self) -> None:
        """Call this immediately before every OpenRouter HTTP request."""
        async with self._lock:
            now = time.monotonic()
            gap = self._last_dispatch + self.min_gap_s - now
            if gap > 0:
                print(f"  [rate-limiter] waiting {gap:.1f}s before next API call…")
                await asyncio.sleep(gap)
            self._last_dispatch = time.monotonic()


# Singleton — created once in main() and passed via closure through openrouter_chat
_rate_limiter: RateLimiter | None = None


# ── Rate-limited OpenRouter helper ─────────────────────────────────────────────

async def openrouter_chat(
    messages: list[dict],
    *,
    label: str,
    logger: LLMCallLogger,
    question_id: str,
    q_idx: int,
    model: str = EVAL_MODEL,
    max_tokens: int = 512,
    judge_correct: Optional[bool] = None,
) -> tuple[str, int]:
    """
    Call OpenRouter /v1/chat/completions. Returns (response_text, prompt_tokens).
    Enforces at least RATE_LIMIT_SLEEP seconds between calls globally via
    RateLimiter, then logs the full input/output to llm_calls.jsonl.
    """
    assert _rate_limiter is not None, "Call main() before openrouter_chat()"
    await _rate_limiter.wait()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            latency = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

            logger.log(
                label=label,
                question_id=question_id,
                q_idx=q_idx,
                model=model,
                messages=messages,
                output=text,
                prompt_tokens=prompt_tokens,
                latency_s=latency,
                correct=judge_correct,
            )
            return text, prompt_tokens

        except Exception as exc:
            wait = (attempt + 1) * 6
            print(
                f"  [warn] API call failed ({exc}), retry {attempt+1}/5 in {wait}s",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)

    logger.log(
        label=f"{label}:ERROR",
        question_id=question_id,
        q_idx=q_idx,
        model=model,
        messages=messages,
        output="[ERROR: all retries failed]",
        prompt_tokens=0,
        latency_s=0.0,
    )
    return "[ERROR: all retries failed]", 0


# ── Context builders ───────────────────────────────────────────────────────────

def format_session(session: list[dict]) -> str:
    lines = []
    for msg in session:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content'].strip()}")
    return "\n\n".join(lines)


# ── A: Qdrant semantic search ──────────────────────────────────────────────────

async def build_qdrant_context(
    qdrant: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    question: str,
    question_id: str,
    top_k: int = QDRANT_TOP_K,
) -> tuple[str, list[dict]]:
    """
    Embed the question, query Qdrant filtered by question_id, return
    (formatted_context_string, raw_chunks_list).

    raw_chunks_list items: {"text": str, "session_id": str, "score": float}
    """
    try:
        query_vector = await embedder.create(question)

        results = await qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="question_id",
                        match=MatchValue(value=question_id),
                    )
                ]
            ),
            limit=top_k,
        )

        if not results.points:
            return "[No relevant context retrieved from vector store]", []

        chunks = []
        formatted_parts = []
        for point in results.points:
            text = point.payload.get("text", "")
            session_id = point.payload.get("session_id", "unknown")
            score = point.score
            chunks.append({"text": text, "session_id": session_id, "score": score})
            formatted_parts.append(
                f"[Session: {session_id} | Score: {score:.4f}]\n{text}"
            )

        context = "SEMANTIC SEARCH RESULTS:\n\n" + "\n\n---\n\n".join(formatted_parts)
        return context, chunks

    except Exception as exc:
        print(
            f"  [warn] Qdrant search failed for {question_id}: {exc}",
            file=sys.stderr,
        )
        return "[Vector store search error]", []


# ── B: Graphiti temporal knowledge graph ───────────────────────────────────────

async def build_graphiti_context_raw(
    graphiti: Graphiti,
    question: str,
    question_id: str,
) -> tuple[list, list]:
    """
    Search Graphiti and return (edges, nodes) as raw objects.
    """
    try:
        results = await graphiti.search(
            query=question,
            group_ids=[question_id],
            num_results=GRAPHITI_TOP_K,
        )

        if hasattr(results, "edges"):
            edges = results.edges or []
            nodes = results.nodes or []
        elif isinstance(results, (list, tuple)) and len(results) == 2:
            edges, nodes = results
        else:
            edges = results if isinstance(results, list) else []
            nodes = []

        return edges, nodes

    except Exception as exc:
        print(
            f"  [warn] Graphiti search failed for {question_id}: {exc}",
            file=sys.stderr,
        )
        return [], []


def format_graphiti_context(edges: list, nodes: list) -> str:
    """Format raw Graphiti edges and nodes into a context string."""
    facts_lines: list[str] = []
    entity_lines: list[str] = []

    for edge in edges:
        fact = getattr(edge, "fact", None) or getattr(edge, "name", "")
        valid_at = getattr(edge, "valid_at", None)
        invalid_at = getattr(edge, "invalid_at", None)
        date_range = ""
        if valid_at or invalid_at:
            date_range = f" (valid: {valid_at or '?'} → {invalid_at or 'present'})"
        facts_lines.append(f"- {fact}{date_range}")

    for node in nodes:
        name = getattr(node, "name", "")
        summary = getattr(node, "summary", "")
        if name:
            entity_lines.append(f"- {name}: {summary}")

    context_parts = []
    if facts_lines:
        context_parts.append("RELEVANT FACTS:\n" + "\n".join(facts_lines))
    if entity_lines:
        context_parts.append("RELEVANT ENTITIES:\n" + "\n".join(entity_lines))

    if not context_parts:
        return "[No relevant context retrieved from knowledge graph]"

    return "\n\n".join(context_parts)


async def build_graphiti_context(
    graphiti: Graphiti,
    question: str,
    question_id: str,
) -> str:
    """Convenience wrapper: fetch + format Graphiti context."""
    edges, nodes = await build_graphiti_context_raw(graphiti, question, question_id)
    return format_graphiti_context(edges, nodes)


# ── C: Hybrid — parallel retrieval + temporal aggregation ──────────────────────

def _extract_expired_entities(edges: list) -> set[str]:
    """
    Return a set of lowercased entity names that appear in *expired* edges
    (i.e. edges where invalid_at is not None).
    """
    expired: set[str] = set()
    for edge in edges:
        invalid_at = getattr(edge, "invalid_at", None)
        if invalid_at is None:
            continue  # still active — skip

        # Try common attribute names for source/target entity names
        for attr in ("source_node_name", "source", "subject"):
            val = getattr(edge, attr, None)
            if val and isinstance(val, str):
                expired.add(val.strip().lower())
                break

        for attr in ("target_node_name", "target", "object"):
            val = getattr(edge, attr, None)
            if val and isinstance(val, str):
                expired.add(val.strip().lower())
                break

        # Also pull entity names from the fact string itself as fallback
        fact = getattr(edge, "fact", None) or getattr(edge, "name", "") or ""
        # Simple heuristic: capitalised words in the fact are likely entity names
        tokens = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", fact)
        for tok in tokens:
            expired.add(tok.lower())

    expired.discard("")
    return expired


def _annotate_chunks(chunks: list[dict], expired_entities: set[str]) -> list[str]:
    """
    For each Qdrant chunk, check if any expired entity name appears in its text.
    Tag expired chunks with a warning; leave valid chunks as-is.
    Returns a list of formatted strings ready for the prompt.
    """
    annotated = []
    for chunk in chunks:
        text = chunk["text"]
        session_id = chunk["session_id"]
        score = chunk["score"]
        text_lower = text.lower()

        is_expired = any(ent in text_lower for ent in expired_entities if ent)

        header = f"[Session: {session_id} | Score: {score:.4f}]"
        if is_expired:
            annotated.append(
                f"⚠️ TEMPORALLY EXPIRED CONTEXT {header}:\n{text}"
            )
        else:
            annotated.append(f"{header}\n{text}")

    return annotated


async def build_hybrid_context(
    qdrant: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    graphiti: Graphiti,
    question: str,
    question_id: str,
) -> str:
    """
    Run Qdrant and Graphiti retrievals in parallel (asyncio.gather).
    Aggregate: cross-reference Qdrant chunks against expired Graphiti edges.
    Chunks whose entities appear in expired edges are tagged ⚠️ TEMPORALLY EXPIRED.
    Returns a unified context string.
    """
    # ── Parallel retrieval ────────────────────────────────────────────────────
    (qdrant_ctx_str, qdrant_chunks), (graph_edges, graph_nodes) = await asyncio.gather(
        build_qdrant_context(qdrant, embedder, question, question_id),
        build_graphiti_context_raw(graphiti, question, question_id),
    )

    # ── Aggregation: tag expired Qdrant chunks ────────────────────────────────
    expired_entities = _extract_expired_entities(graph_edges)
    annotated_chunks = _annotate_chunks(qdrant_chunks, expired_entities)

    # ── Build unified prompt context ──────────────────────────────────────────
    parts = []

    if annotated_chunks:
        chunks_block = "\n\n---\n\n".join(annotated_chunks)
        expired_count = sum(1 for c in annotated_chunks if c.startswith("⚠️"))
        note = (
            f"  [{expired_count}/{len(annotated_chunks)} chunks flagged as temporally expired]"
            if expired_count > 0
            else ""
        )
        parts.append(f"SEMANTIC SEARCH RESULTS:{note}\n\n{chunks_block}")
    else:
        parts.append("SEMANTIC SEARCH RESULTS:\n[No results from vector store]")

    graph_ctx = format_graphiti_context(graph_edges, graph_nodes)
    parts.append(graph_ctx)

    return "\n\n" + ("=" * 60) + "\n\n".join(parts)


# ── LLM prompts ────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are a helpful AI assistant with access to a memory of past conversations. "
    "Answer the user's question based on the context provided. "
    "Be concise and direct. If the answer is not in the context, say 'I don't know'. "
    "If context is marked ⚠️ TEMPORALLY EXPIRED, treat it as outdated — "
    "prefer non-expired context or Graphiti facts when they conflict."
)


def answer_prompt(context: str, question: str, question_date: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Context from past conversations (as of {question_date}):\n\n"
                f"{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]


JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether an AI assistant's answer "
    "is correct given a ground-truth reference answer. "
    "Reply with ONLY 'CORRECT' or 'INCORRECT', followed by a one-sentence reason."
)


def judge_prompt(question: str, ground_truth: str, model_answer: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Ground Truth Answer: {ground_truth}\n\n"
                f"Model Answer: {model_answer}\n\n"
                "Is the model answer correct? Reply CORRECT or INCORRECT."
            ),
        },
    ]


def parse_judge_verdict(judge_response: str) -> bool:
    return judge_response.strip().upper().startswith("CORRECT")


# ── Single question evaluator ──────────────────────────────────────────────────

async def evaluate_question(
    qdrant: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    graphiti: Graphiti,
    case: dict,
    q_idx: int,
    logger: LLMCallLogger,
) -> dict:
    """
    Evaluate one question under all three approaches (Qdrant, Graphiti, Hybrid).
    Returns a result dict with metrics for all three.
    """
    question_id = case["question_id"]
    question_type = case["question_type"]
    question_date = case["question_date"]
    question = case["questions"][q_idx]
    ground_truth = case["answers"][q_idx]

    result = {
        "question_id": question_id,
        "question_type": question_type,
        "question_date": question_date,
        "question_index": q_idx,
        "question": question,
        "ground_truth": ground_truth,
    }

    # Shared kwargs for openrouter_chat
    ctx = dict(logger=logger, question_id=question_id, q_idx=q_idx)

    async def run_approach(
        approach_label: str,
        context_str: str,
        context_snippet: str,
    ) -> dict:
        """
        Given a pre-built context string, call the LLM for an answer,
        then call the judge. Returns a dict of metrics for this approach.
        """
        prefix = approach_label  # "qdrant" | "graph" | "hybrid"

        try:
            messages = answer_prompt(context_str, question, question_date)
            t0 = time.perf_counter()
            answer, prompt_tokens = await openrouter_chat(
                messages, label=f"{prefix}_answer", max_tokens=512, **ctx
            )
            latency = time.perf_counter() - t0

            judge_msgs = judge_prompt(question, ground_truth, answer)
            judge_resp, _ = await openrouter_chat(
                judge_msgs, label=f"{prefix}_judge", max_tokens=64, **ctx
            )
            correct = parse_judge_verdict(judge_resp)

            # Re-log verdict with correct flag
            logger.log(
                label=f"{prefix}_judge:verdict",
                question_id=question_id,
                q_idx=q_idx,
                model=EVAL_MODEL,
                messages=judge_msgs,
                output=judge_resp,
                prompt_tokens=0,
                latency_s=0.0,
                correct=correct,
            )

            return {
                f"{prefix}_context_snippet": context_snippet,
                f"{prefix}_answer": answer,
                f"{prefix}_correct": correct,
                f"{prefix}_latency_s": round(latency, 3),
                f"{prefix}_prompt_tokens": prompt_tokens,
                f"{prefix}_judge_response": judge_resp,
            }

        except Exception as exc:
            print(f"  [error] {prefix} eval failed: {exc}", file=sys.stderr)
            return {
                f"{prefix}_context_snippet": "",
                f"{prefix}_answer": f"[ERROR: {exc}]",
                f"{prefix}_correct": False,
                f"{prefix}_latency_s": 0.0,
                f"{prefix}_prompt_tokens": 0,
                f"{prefix}_judge_response": "",
            }

    # ── Build all three contexts ───────────────────────────────────────────────
    # Qdrant and Graphiti raw contexts are also needed for hybrid, but we avoid
    # triple fetching by building hybrid independently (it fetches internally).
    # Qdrant and Graphiti standalone approaches fetch separately — acceptable
    # because the goal is fair isolated measurement per approach.

    qdrant_ctx, _ = await build_qdrant_context(qdrant, embedder, question, question_id)
    graph_ctx = await build_graphiti_context(graphiti, question, question_id)
    hybrid_ctx = await build_hybrid_context(qdrant, embedder, graphiti, question, question_id)

    # ── Run approaches sequentially to respect rate limits ────────────────────
    # (parallel retrieval is INSIDE build_hybrid_context; LLM calls are sequential
    #  to avoid bursting the OpenRouter 20-RPM free-tier limit)
    qdrant_metrics = await run_approach(
        "qdrant", qdrant_ctx,
        qdrant_ctx[:300].replace("\n", " ") + "…",
    )
    graph_metrics = await run_approach(
        "graph", graph_ctx,
        graph_ctx[:300].replace("\n", " ") + "…",
    )
    hybrid_metrics = await run_approach(
        "hybrid", hybrid_ctx,
        hybrid_ctx[:300].replace("\n", " ") + "…",
    )

    result.update(qdrant_metrics)
    result.update(graph_metrics)
    result.update(hybrid_metrics)

    # ── Console summary ───────────────────────────────────────────────────────
    def v(key): return "✓" if result.get(key) else "✗"
    def l(key): return f"{result.get(key, 0.0):.1f}s"

    print(
        f"      Q{q_idx}: "
        f"Qdrant={v('qdrant_correct')}({l('qdrant_latency_s')}) | "
        f"Graph={v('graph_correct')}({l('graph_latency_s')}) | "
        f"Hybrid={v('hybrid_correct')}({l('hybrid_latency_s')})"
    )

    return result


# ── Summary table builders ─────────────────────────────────────────────────────

def compute_tables(all_results: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (table2_data, table3_data).
    Table 2: overall metrics for all three approaches.
    Table 3: per question-type breakdown.
    """
    APPROACHES = [
        ("qdrant",  "Qdrant (Semantic)"),
        ("graph",   "Graphiti (Graph)"),
        ("hybrid",  "Hybrid (Parallel)"),
    ]

    # ── Table 2: overall ──────────────────────────────────────────────────────
    def summarise(results: list[dict], prefix: str, label: str) -> dict:
        correct_key = f"{prefix}_correct"
        latency_key = f"{prefix}_latency_s"
        tokens_key  = f"{prefix}_prompt_tokens"

        valid = [r for r in results if r.get(correct_key) is not None]
        if not valid:
            return {}

        n = len(valid)
        n_correct = sum(1 for r in valid if r[correct_key])
        latencies = sorted(r[latency_key] for r in valid)
        tokens = [r[tokens_key] for r in valid]

        q1 = latencies[int(n * 0.25)]
        q3 = latencies[min(int(n * 0.75), n - 1)]

        return {
            "approach": label,
            "model": EVAL_MODEL,
            "n_questions": n,
            "score_pct": round(100 * n_correct / n, 1),
            "mean_latency_s": round(statistics.mean(latencies), 3),
            "latency_iqr_s": round(q3 - q1, 3),
            "avg_prompt_tokens": round(statistics.mean(tokens)),
        }

    table2 = [summarise(all_results, prefix, label) for prefix, label in APPROACHES]

    # ── Table 3: per question-type ────────────────────────────────────────────
    q_types = sorted({r["question_type"] for r in all_results})
    table3 = []
    for qt in q_types:
        subset = [r for r in all_results if r["question_type"] == qt]
        n = len(subset)

        def pct(key):
            return round(100 * sum(1 for r in subset if r.get(key)) / n, 1)

        qdrant_pct = pct("qdrant_correct")
        graph_pct  = pct("graph_correct")
        hybrid_pct = pct("hybrid_correct")

        best = max(qdrant_pct, graph_pct, hybrid_pct)
        best_label = (
            "Hybrid" if hybrid_pct == best
            else "Graphiti" if graph_pct == best
            else "Qdrant"
        )

        table3.append({
            "question_type": qt,
            "model": EVAL_MODEL,
            "n": n,
            "qdrant_pct":  qdrant_pct,
            "graphiti_pct": graph_pct,
            "hybrid_pct":  hybrid_pct,
            "delta_hybrid_vs_qdrant":  round(hybrid_pct - qdrant_pct, 1),
            "delta_hybrid_vs_graphiti": round(hybrid_pct - graph_pct, 1),
            "best_approach": best_label,
        })

    return table2, table3


# ── Console printers ───────────────────────────────────────────────────────────

def print_table2(table2: list[dict]) -> None:
    W = 88
    print("\n" + "=" * W)
    print("TABLE 2 — Overall Results (Three-Way Comparison)")
    print("=" * W)
    print(
        f"{'Approach':<22} {'Model':<28} {'Score':>7} "
        f"{'Lat(s)':>8} {'IQR(s)':>7} {'Tokens':>8}"
    )
    print("-" * W)
    for row in table2:
        if not row:
            continue
        print(
            f"{row['approach']:<22} {row['model']:<28} "
            f"{row['score_pct']:>6.1f}% {row['mean_latency_s']:>7.2f} "
            f"{row['latency_iqr_s']:>6.3f} {row['avg_prompt_tokens']:>8}"
        )
    print("=" * W)


def print_table3(table3: list[dict]) -> None:
    W = 88
    print("\n" + "=" * W)
    print("TABLE 3 — Per Question-Type Breakdown")
    print("=" * W)
    print(
        f"{'Question Type':<28} {'n':>4} {'Qdrant%':>8} "
        f"{'Graphiti%':>10} {'Hybrid%':>8} {'Best':>10}"
    )
    print("-" * W)
    for row in table3:
        print(
            f"{row['question_type']:<28} {row['n']:>4} "
            f"{row['qdrant_pct']:>7.1f}% {row['graphiti_pct']:>9.1f}% "
            f"{row['hybrid_pct']:>7.1f}% {row['best_approach']:>10}"
        )
    print("=" * W)


# ── CSV writers ────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Global rate limiter (must be created inside the running event loop) ────
    global _rate_limiter
    _rate_limiter = RateLimiter(min_gap_s=RATE_LIMIT_SLEEP)
    print(f"Rate limiter active: ≥{RATE_LIMIT_SLEEP}s between every OpenRouter call.\n")

    # ── LLM call logger ────────────────────────────────────────────────────────
    log_path = RESULTS_DIR / "llm_calls.jsonl"
    logger = LLMCallLogger(log_path)
    print(f"LLM call log → {log_path}  (tail -f to watch live)\n")

    # ── Load benchmark ─────────────────────────────────────────────────────────
    print(f"Loading {JSON_PATH}…")
    with open(JSON_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    if MAX_CASES is not None:
        data = data[:MAX_CASES]
    print(f"Evaluating {len(data)} cases.\n")

    # ── Init Qdrant ────────────────────────────────────────────────────────────
    print(f"Connecting to Qdrant at {QDRANT_URL}…")
    qdrant = AsyncQdrantClient(url=QDRANT_URL, timeout=10.0)
    info = await qdrant.get_collection(QDRANT_COLLECTION)
    print(f"  Collection '{QDRANT_COLLECTION}': {info.points_count} points indexed.\n")

    # ── Init embedder ──────────────────────────────────────────────────────────
    print(f"Loading embedder ({EMBED_MODEL})…")
    embedder = SentenceTransformerEmbedder(model_name=EMBED_MODEL)
    print("  Embedder ready.\n")

    # ── Init Graphiti ──────────────────────────────────────────────────────────
    print("Connecting to Neo4j / Graphiti…")
    llm_client = OpenAIClient(
        config=LLMConfig(
            api_key=OPENROUTER_API_KEY,
            model=GRAPH_LLM_MODEL,
            base_url=OPENROUTER_BASE_URL,
        )
    )
    graphiti = Graphiti(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
    )
    print("  Graphiti ready.\n")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    total = len(data)

    try:
        for case_idx, case in enumerate(data, start=1):
            qid   = case["question_id"]
            qtype = case["question_type"]
            n_qs  = len(case["questions"])
            print(f"[{case_idx}/{total}] {qid}  type={qtype}  questions={n_qs}")

            for q_idx in range(n_qs):
                result = await evaluate_question(
                    qdrant, embedder, graphiti, case, q_idx, logger
                )
                all_results.append(result)

            # Checkpoint after every case
            raw_path = RESULTS_DIR / "raw_results.json"
            with open(raw_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    finally:
        logger.close()

    # ── Compute & print tables ─────────────────────────────────────────────────
    print("\nComputing summary tables…")
    table2, table3 = compute_tables(all_results)

    print_table2(table2)
    print_table3(table3)

    # Save CSV + JSON
    save_csv(table2, RESULTS_DIR / "summary_table2.csv")
    save_csv(table3, RESULTS_DIR / "summary_table3.csv")

    with open(RESULTS_DIR / "summary_table2.json", "w") as f:
        json.dump(table2, f, indent=2)
    with open(RESULTS_DIR / "summary_table3.json", "w") as f:
        json.dump(table3, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print(f"  llm_calls.jsonl        (live log)")
    print(f"  raw_results.json       ({len(all_results)} question evaluations)")
    print(f"  summary_table2.csv/json  (overall)")
    print(f"  summary_table3.csv/json  (per question-type)")

    await graphiti.close()
    print("\nDone ✓")


if __name__ == "__main__":
    asyncio.run(main())
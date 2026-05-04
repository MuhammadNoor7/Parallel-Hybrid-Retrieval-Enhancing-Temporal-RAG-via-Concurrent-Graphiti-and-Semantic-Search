"""
LongMemEval → Graphiti ingestion (MapReduce Architecture)
=========================================================
Reads longmemeval_s.json and writes every haystack conversation session
into a Graphiti knowledge graph backed by Neo4j.

ARCHITECTURE: MAP-REDUCE
To prevent temporal corruption while maximizing API concurrency, this script
implements a custom MapReduce pipeline:
  1. MAP (Parallel)   : Raw conversations are pre-summarized into facts via concurrent LLM calls.
  2. SORT             : Extracted facts are sorted chronologically by timestamp.
  3. REDUCE (Sequence): Sorted facts are fed into Graphiti sequentially to preserve the timeline.

LLM  : Gemma 4 via OpenRouter
Embed: Local sentence-transformers model (BAAI/bge-base-en-v1.5) on GPU/CPU
Graph: Neo4j running locally
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType

from local_embedder import SentenceTransformerEmbedder

load_dotenv()

# ── LLM: OpenRouter Configuration ─────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL: str = "inclusionai/ling-2.6-1t:free"
SMALL_MODEL: str = "google/gemma-4-26b-a4b-it"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Embedder: local sentence-transformers on your GTX 1070 ────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# ── Neo4j (local) ─────────────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "password")

# ── Ingestion knobs ────────────────────────────────────────────────────────────
JSON_PATH: str = os.getenv("JSON_PATH", "data/single-session-preference.json")
MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "5"))  # Max parallel API calls
PROGRESS_FILE: str = os.getenv("PROGRESS_FILE", "progress.json")

ENABLE_SELECTIVE_THROTTLING: bool = True 


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(path: str) -> set[str]:
    """Return the set of session IDs that have already been ingested."""
    try:
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def save_progress(path: str, completed: set[str]) -> None:
    """Persist the completed-session set to disk (atomic-ish via temp file)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(completed), f)
    os.replace(tmp, path)  


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2})\s+\([A-Za-z]+\)\s+(\d{2}:\d{2})")

def parse_haystack_date(date_str: str) -> datetime:
    """Parse a LongMemEval date string into an aware UTC datetime."""
    m = _DATE_RE.search(date_str)
    if m:
        return datetime.strptime(
            f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M"
        ).replace(tzinfo=timezone.utc)
    print(f"  [warn] Could not parse date '{date_str}', using now()", file=sys.stderr)
    return datetime.now(timezone.utc)


def format_conversation(session: list[dict]) -> str:
    """Convert a haystack session into a plain-text transcript."""
    lines: list[str] = []
    for msg in session:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content'].strip()}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAP-REDUCE Core Logic
# ─────────────────────────────────────────────────────────────────────────────

async def map_extract_facts(
    session_id: str,
    session: list[dict],
    ref_time: datetime,
    raw_openai_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> tuple[datetime, str, str]:
    """
    MAP PHASE (Parallel):
    Offloads heavy LLM extraction from Graphiti. We run raw API calls concurrently 
    to extract facts and entities from the conversation text before ingestion.
    """
    conversation_text = format_conversation(session)
    prompt = (
        "You are a data extraction assistant. Summarize the following conversation "
        "into a concise, structured list of standalone facts, entity names, and events. "
        "This text will be ingested directly into a temporal knowledge graph.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    async with semaphore:
        try:
            response = await raw_openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            extracted_text = response.choices[0].message.content
            return (ref_time, session_id, extracted_text)
        except Exception as e:
            print(f"    [Map Error] session {session_id}: {e}")
            # Fallback to the raw conversation text if extraction fails
            return (ref_time, session_id, conversation_text)


async def reduce_ingest_session(
    graphiti: Graphiti,
    *,
    question_id: str,
    session_id: str,
    episode_body: str,
    ref_time: datetime,
    semaphore: asyncio.Semaphore,
    completed: set[str],
    progress_lock: asyncio.Lock,
) -> None:
    """
    REDUCE PHASE (Sequential): 
    Inserts the PRE-EXTRACTED facts into Graphiti. By running this sequentially,
    we ensure Graphiti resolves temporal contradictions perfectly.
    """
    if session_id in completed:
        print(f"    ↷  session {session_id} (already ingested, skipping)")
        return

    async with semaphore:
        try:
            await graphiti.add_episode(
                name=session_id,
                episode_body=episode_body, # Passing the summarized facts, not raw text
                source_description=(
                    "LongMemEval haystack conversation session — "
                    f"evaluation case {question_id}"
                ),
                reference_time=ref_time,
                source=EpisodeType.message,
                group_id=question_id,
            )

            async with progress_lock:
                completed.add(session_id)
                save_progress(PROGRESS_FILE, completed)

            print(f"    ✓  session {session_id}")
        except Exception as exc:
            print(f"    ✗  session {session_id}: {exc}", file=sys.stderr)


async def ingest_case(
    graphiti: Graphiti,
    case: dict,
    raw_openai_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total: int,
    completed: set[str],
    progress_lock: asyncio.Lock,
) -> None:
    """Executes the MapReduce pipeline for a single evaluation case."""
    question_id = case["question_id"]
    sessions = case["haystack_sessions"]
    session_ids = case["haystack_session_ids"]
    haystack_dates = case["haystack_dates"]
    n_sessions = len(sessions)

    print(f"\n[{case_index}/{total}] {question_id} ({n_sessions} sessions)")
    
    # --- 1. MAP (Concurrent Extraction) ---
    print(f"  └── 1. MAP: Extracting facts concurrently...")
    map_tasks = [
        map_extract_facts(sid, sess, parse_haystack_date(date_str), raw_openai_client, semaphore)
        for sid, sess, date_str in zip(session_ids, sessions, haystack_dates)
    ]
    mapped_results = await asyncio.gather(*map_tasks)

    # --- 2. SORT (Chronological Ordering) ---
    print(f"  └── 2. SORT: Ordering chronologically to protect timeline...")
    mapped_results.sort(key=lambda x: x[0])

    # --- 3. REDUCE (Sequential Graphiti Ingestion) ---
    print(f"  └── 3. REDUCE: Sequentially inserting facts into Graphiti...")
    for ref_time, sid, extracted_text in mapped_results:
        await reduce_ingest_session(
            graphiti,
            question_id=question_id,
            session_id=sid,
            episode_body=extracted_text,
            ref_time=ref_time,
            semaphore=semaphore,
            completed=completed,
            progress_lock=progress_lock,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    
    if ENABLE_SELECTIVE_THROTTLING:
        # ── Custom HTTP Interceptor for Selective Rate Limiting ────────────────
        http_client = httpx.AsyncClient()
        original_send = http_client.send

        async def throttled_send(request: httpx.Request, *args, **kwargs):
            if "chat/completions" in str(request.url) and request.content:
                try:
                    body = json.loads(request.content.decode("utf-8"))
                    is_structured = "response_format" in body or "tools" in body
                    if not is_structured:
                        await asyncio.sleep(3.0)
                except Exception:
                    pass
            return await original_send(request, *args, **kwargs)

        http_client.send = throttled_send

        raw_openai_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            http_client=http_client
        )
    else:
        raw_openai_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    # ── Build Graphiti clients ─────────────────────────────────────────────
    llm_client = OpenAIClient(
        config=LLMConfig(
            api_key=OPENROUTER_API_KEY,
            model=LLM_MODEL,
            small_model=SMALL_MODEL,
            base_url=OPENROUTER_BASE_URL,
        ),
        client=raw_openai_client 
    )

    embedder = SentenceTransformerEmbedder(model_name=EMBED_MODEL)

    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
    )

    print("Setting up Neo4j indices and constraints…")
    await graphiti.build_indices_and_constraints()

    print(f"Loading {JSON_PATH}…")
    with open(JSON_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    total = len(data)
    print(f"Loaded {total} evaluation cases.\n")

    completed: set[str] = load_progress(PROGRESS_FILE)
    if completed:
        print(f"Resuming — {len(completed)} session(s) already ingested.\n")
    progress_lock = asyncio.Lock()

    # Semaphore bounds the total number of simultaneous API calls across both Map & Reduce
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # We can run multiple EVALUATION CASES concurrently, and each case will run its own MapReduce!
    await asyncio.gather(
        *[
            ingest_case(graphiti, case, raw_openai_client, semaphore, i, total, completed, progress_lock)
            for i, case in enumerate(data, start=1)
        ]
    )

    print("\nAll cases ingested. Closing connection…")
    await graphiti.close()
    print("Done ✓")


if __name__ == "__main__":
    asyncio.run(main())
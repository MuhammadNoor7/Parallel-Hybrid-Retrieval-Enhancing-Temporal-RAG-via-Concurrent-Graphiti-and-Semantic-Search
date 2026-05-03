"""
LongMemEval → Graphiti ingestion
=================================
Reads longmemeval_s.json and writes every haystack conversation session
into a Graphiti knowledge graph backed by Neo4j.

Each episode represents one full haystack session (conversation).
All sessions that belong to the same evaluation case share a `group_id`
equal to that case's `question_id`, so you can query by case later.

LLM  : MiniMax M2.5 via OpenRouter
Embed: Local sentence-transformers model (BAAI/bge-base-en-v1.5) on GPU/CPU
Graph: Neo4j running locally
"""

import time
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType

from local_embedder import SentenceTransformerEmbedder

load_dotenv()

# ── LLM: MiniMax M2.5 via OpenRouter ──────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL: str = "inclusionai/ling-2.6-1t:free"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Embedder: local sentence-transformers on your GTX 1070 ────────────────────
#   BAAI/bge-base-en-v1.5  → 768-dim, great quality, recommended
#   BAAI/bge-small-en-v1.5 → 384-dim, fastest
#   BAAI/bge-large-en-v1.5 → 1024-dim, best quality, still fits on 8 GB VRAM
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# ── Neo4j (local) ─────────────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ["NEO4J_PASSWORD"]

# ── Ingestion knobs ────────────────────────────────────────────────────────────
JSON_PATH: str = os.getenv("JSON_PATH", "data/longmemeval_s.json")
MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "1"))  # parallel add_episode calls


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches "2023/12/18 (Mon) 04:17"
_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2})\s+\([A-Za-z]+\)\s+(\d{2}:\d{2})")


def parse_haystack_date(date_str: str) -> datetime:
    """Parse a LongMemEval date string into an aware UTC datetime."""
    m = _DATE_RE.search(date_str)
    if m:
        return datetime.strptime(
            f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M"
        ).replace(tzinfo=timezone.utc)
    # Fallback: current time so the episode still ingests cleanly
    print(f"  [warn] Could not parse date '{date_str}', using now()", file=sys.stderr)
    return datetime.now(timezone.utc)


def format_conversation(session: list[dict]) -> str:
    """
    Convert a haystack session (list of {role, content} dicts) into a
    plain-text transcript that Graphiti can extract entities from.

    Example output:
        User: What's a good pasta recipe?

        Assistant: Here's a simple carbonara …
    """
    lines: list[str] = []
    for msg in session:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content'].strip()}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core ingestion logic
# ─────────────────────────────────────────────────────────────────────────────


async def ingest_session(
    graphiti: Graphiti,
    *,
    question_id: str,
    session_id: str,
    session: list[dict],
    date_str: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Add a single haystack session as a Graphiti episode."""
    episode_body = format_conversation(session)
    ref_time = parse_haystack_date(date_str)

    async with semaphore:
        try:
            # Add a 4-second delay before the graphiti call 
            # to stay safely under OpenRouter's 20 RPM free-tier limit.
            await asyncio.sleep(4)
            
            await graphiti.add_episode(
                name=session_id,
                episode_body=episode_body,
                source_description=(
                    "LongMemEval haystack conversation session — "
                    f"evaluation case {question_id}"
                ),
                reference_time=ref_time,
                source=EpisodeType.message,
                # group_id ties all sessions from the same eval case together,
                # which lets you scope graph queries by question_id later.
                group_id=question_id,
            )
            print(f"    ✓  session {session_id}")
        except Exception as exc:
            print(f"    ✗  session {session_id}: {exc}", file=sys.stderr)


async def ingest_case(
    graphiti: Graphiti,
    case: dict,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total: int,
) -> None:
    """Ingest all haystack sessions that belong to one evaluation case."""
    question_id = case["question_id"]
    sessions = case["haystack_sessions"]
    session_ids = case["haystack_session_ids"]
    haystack_dates = case["haystack_dates"]

    n_sessions = len(sessions)
    print(f"[{case_index}/{total}] {question_id}  ({n_sessions} sessions)")

    tasks = [
        ingest_session(
            graphiti,
            question_id=question_id,
            session_id=sid,
            session=sess,
            date_str=date_str,
            semaphore=semaphore,
        )
        for sid, sess, date_str in zip(session_ids, sessions, haystack_dates)
    ]

    # Run sessions for this case concurrently (bounded by semaphore)
    await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    # ── Build Graphiti clients ─────────────────────────────────────────────
    llm_client = OpenAIClient(
        config=LLMConfig(
            api_key=OPENROUTER_API_KEY,
            model=LLM_MODEL,
            base_url=OPENROUTER_BASE_URL,
        )
    )

    # Local embedder — downloads model on first run, then caches to ~/.cache/huggingface
    embedder = SentenceTransformerEmbedder(model_name=EMBED_MODEL)

    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
    )

    # ── One-time Neo4j setup (idempotent) ──────────────────────────────────
    print("Setting up Neo4j indices and constraints…")
    await graphiti.build_indices_and_constraints()

    # ── Load JSON ──────────────────────────────────────────────────────────
    print(f"Loading {JSON_PATH}…")
    with open(JSON_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    total = len(data)
    print(f"Loaded {total} evaluation cases.\n")

    # ── Ingest ────────────────────────────────────────────────────────────
    # Cases are processed sequentially; sessions within each case run
    # concurrently up to MAX_CONCURRENT to stay within rate limits.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    for i, case in enumerate(data, start=1):
        await ingest_case(graphiti, case, semaphore, i, total)
        time.sleep(3)

    print("\nAll cases ingested. Closing connection…")
    await graphiti.close()
    print("Done ✓")


if __name__ == "__main__":
    asyncio.run(main())
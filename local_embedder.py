"""
local_embedder.py
-----------------
A Graphiti-compatible EmbedderClient backed by a local sentence-transformers
model running on your GPU (or CPU as fallback).

Recommended model for a GTX 1070 / 8 GB VRAM:
    BAAI/bge-base-en-v1.5  → 768-dim, fast, high quality

Other options (uncomment in ingest.py):
    BAAI/bge-small-en-v1.5  → 384-dim, even faster, slightly lower quality
    BAAI/bge-large-en-v1.5  → 1024-dim, best quality, still fits on a 1070
    nomic-ai/nomic-embed-text-v1 → 768-dim, needs trust_remote_code=True
"""

import asyncio
from collections.abc import Iterable
from functools import partial

import torch
from sentence_transformers import SentenceTransformer

from graphiti_core.embedder.client import EmbedderClient


class SentenceTransformerEmbedder(EmbedderClient):
    """
    Wraps a sentence-transformers model as a Graphiti EmbedderClient.

    The model's encode() call is synchronous and potentially slow, so it is
    dispatched to a thread-pool executor to avoid blocking the event loop.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LocalEmbedder] Loading '{model_name}' on {device.upper()}…")
        self._model = SentenceTransformer(model_name, device=device)
        self._loop = asyncio.get_event_loop()
        print(f"[LocalEmbedder] Ready — embedding dim: {self._model.get_sentence_embedding_dimension()}")

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Blocking encode — runs in a thread-pool so the event loop stays free."""
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity works out of the box
            show_progress_bar=False,
        )
        return vecs.tolist()

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        """Embed a single string (Graphiti's primary call path)."""
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            # Graphiti may pass token-id iterables in some paths; convert to str
            texts = [str(input_data)]

        result = await self._loop.run_in_executor(None, partial(self._encode_sync, texts))
        return result[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed a batch of strings — more efficient than calling create() in a loop."""
        return await self._loop.run_in_executor(
            None, partial(self._encode_sync, input_data_list)
        )
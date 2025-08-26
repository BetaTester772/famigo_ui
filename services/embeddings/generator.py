from __future__ import annotations

"""Embedder factory.

Usage
-----
from app.services.embeddings.generator import get_embedder
embed = get_embedder()
vec = embed("안녕, 테스트 문장")  # -> List[float] length matches VECTOR_DIM

Providers
---------
- OPENAI: set ENV `EMBED_PROVIDER=openai`, optionally `EMBED_MODEL=text-embedding-3-small`
- LOCAL  : default fallback; deterministic hash-based unit vector (no deps)
"""

import os
import math
import hashlib
from typing import Callable, List

try:
    # keep import optional
    from db.models import VECTOR_DIM
except Exception:
    VECTOR_DIM = 1536


# ----------------------------- Local stub ---------------------------------

def _hash_to_unit_vector(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """Deterministic, dependency-free pseudo-embedding.
    Not semantically meaningful but non-zero and stable. Good for wiring/tests.
    """
    vec = [0.0] * dim
    if not text:
        vec[0] = 1.0
        return vec
    for tok in text.split():
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # use chunks of 4 bytes to get multiple positions
        for i in range(0, min(len(h), 32), 4):
            idx = int.from_bytes(h[i:i+4], "big") % dim
            val = ((h[i] / 255.0) * 2.0) - 1.0  # [-1, 1)
            vec[idx] += val
    # L2 normalize
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ----------------------------- OpenAI provider ----------------------------

def _openai_embedder(model: str) -> Callable[[str], List[float]]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai package not installed. `pip install openai`.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)

    def _embed(text: str) -> List[float]:
        if not text:
            text = " "
        out = client.embeddings.create(model=model, input=text)
        return list(out.data[0].embedding)

    return _embed


# ----------------------------- Factory ------------------------------------

def get_embedder() -> Callable[[str], List[float]]:
    provider = os.getenv("EMBED_PROVIDER", "local").lower()
    if provider == "openai":
        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        return _openai_embedder(model)
    # default local stub
    return lambda text: _hash_to_unit_vector(text, VECTOR_DIM)

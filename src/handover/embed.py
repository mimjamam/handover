"""Turn ticket text into vectors with a small multilingual model on CPU."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_TOKENS = 256  # longer tickets are cut; keeps CPU time predictable


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_TOKENS
    return model


def embed(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    """Return one unit-length vector per text.

    The e5 models expect a task prefix; "query: " is the one for
    classification-style use.
    """
    prefixed = [f"query: {t}" for t in texts]
    vectors = _model().encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
    )
    return np.asarray(vectors, dtype=np.float32)

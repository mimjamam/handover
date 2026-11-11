"""Turn ticket text into vectors with a small multilingual model on CPU.

The model is multilingual-e5-small in its 8-bit quantised ONNX form: about
118 MB instead of 470 MB, and it runs on ONNX Runtime without PyTorch.
That is what lets the whole service fit in 512 MB of memory. The same
model is used for building the memory and for answering requests, so the
vectors are always comparable.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
ONNX_REPO = "Xenova/multilingual-e5-small"
ONNX_FILE = "onnx/model_quantized.onnx"
MAX_TOKENS = 256  # longer tickets are cut; keeps CPU time predictable


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    # The plain SentencePiece model needs about 50 MB of memory; the
    # `tokenizers` version of the same vocabulary needs about 300 MB.
    pieces = spm.SentencePieceProcessor(model_file=hf_hub_download(MODEL_NAME, "sentencepiece.bpe.model"))

    options = ort.SessionOptions()
    if os.environ.get("HANDOVER_LOW_MEMORY") == "1":
        # Small hosts: give memory back after each request and use few threads.
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        hf_hub_download(ONNX_REPO, ONNX_FILE), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_names = {i.name for i in session.get_inputs()}
    return pieces, session, input_names


def token_ids(text: str) -> list[int]:
    """Token numbers in the layout the model was trained with (XLM-RoBERTa).

    SentencePiece numbers are shifted by one; 0 and 2 mark start and end,
    3 is "unknown". Long texts are cut to MAX_TOKENS.
    """
    pieces, _, _ = _load()
    body = [p + 1 if p != 0 else 3 for p in pieces.encode(text)][: MAX_TOKENS - 2]
    return [0, *body, 2]


def _encode_one(text: str) -> np.ndarray:
    _, session, input_names = _load()
    ids = np.array([token_ids(text)], dtype=np.int64)
    mask = np.ones_like(ids)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    hidden = session.run(None, feed)[0][0]  # one vector per token
    pooled = hidden.mean(axis=0)
    return pooled / np.linalg.norm(pooled)


def embed(texts: list[str], progress: bool = False) -> np.ndarray:
    """Return one unit-length vector per text.

    Texts are processed one at a time on purpose. The quantised model picks
    its number ranges from whatever is in the batch, so the same text gets a
    slightly different vector depending on its batch neighbours and padding
    (cosine about 0.995 instead of 1.0). That is enough to move a ticket
    across a similarity threshold. One at a time, a text always gets the
    same vector, in training and in the running service alike.

    The e5 models expect a task prefix; "query: " is the one for
    classification-style use.
    """
    out = np.empty((len(texts), 384), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = _encode_one(f"query: {text}")
        if progress and i % 2000 == 0:
            print(f"embedded {i}/{len(texts)}", flush=True)
    return out

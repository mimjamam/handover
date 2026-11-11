import numpy as np

from handover.embed import embed

TEXTS = [
    "Meine Rechnung wurde doppelt abgebucht.",
    "The VPN keeps disconnecting on my laptop since this morning. I already restarted the router twice.",
    "Hallo",
]


def test_vectors_have_unit_length():
    v = embed(TEXTS)
    assert v.shape == (3, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_a_text_gets_the_same_vector_alone_or_in_a_group():
    """Training embeds many texts, the service embeds one. Both must agree exactly."""
    together = embed(TEXTS)
    alone = np.vstack([embed([t]) for t in TEXTS])
    assert np.allclose(together, alone, atol=1e-6)
    assert np.allclose(embed(TEXTS[::-1])[::-1], together, atol=1e-6)

"""The routing model: route a new ticket like the most similar past ticket.

How a decision is made:
  1. Find the K most similar past tickets (cosine similarity of embeddings).
  2. Propose the team of the single most similar one.
  3. Estimate the chance that this proposal is right from three signals:
     how similar the closest ticket is, how many of the K neighbours agree
     with it, and how tightly the K neighbours are packed.
  4. Answer if that chance is above the threshold; otherwise hand over.
  5. Novelty gate: if even the closest past ticket is not similar enough,
     hand over no matter what step 3 says. The confidence model was only
     ever shown tickets like the past ones, so it cannot be trusted on a
     message unlike anything in memory.

Everything the service needs lives in one `Router` object saved to one
file. Training happens offline (scripts/train.py); the service only loads.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ARTIFACT_PATH = Path("artifacts/router.joblib")
K = 5


def confidence_features(top_sim: np.ndarray, agreement: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """Scale the three signals to similar ranges for the confidence model."""
    return np.c_[(1.0 - top_sim) * 100.0, agreement, spread * 100.0]


def fit_confidence_model(features: np.ndarray, correct: np.ndarray) -> LogisticRegression:
    """Learn P(proposal is right) from the three signals.

    Fitted on held-out tickets, so the output is a calibrated probability
    and not just a score.
    """
    return LogisticRegression(C=1.0, max_iter=2000).fit(features, correct)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Average gap between stated confidence and actual accuracy, over confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


@dataclass
class Router:
    classes: list[str]
    index: np.ndarray  # past-ticket embeddings, one unit-length row each
    index_labels: np.ndarray  # team of each past ticket, as an index into `classes`
    index_texts: list[str]  # text of each past ticket, shown as the reason for a decision
    conf_coef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    conf_intercept: float = 0.0
    threshold: float = 1.0  # answer only if confidence >= threshold
    novelty_gate: float = 0.0  # and only if the closest past ticket is at least this similar
    metadata: dict = field(default_factory=dict)

    def neighbours(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions and similarities of the K most similar past tickets, best first."""
        sims = X.astype(np.float32) @ self.index.astype(np.float32).T
        idx = np.argpartition(-sims, K, axis=1)[:, :K]
        top = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-top, axis=1)
        return np.take_along_axis(idx, order, axis=1), np.take_along_axis(top, order, axis=1)

    def propose(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Proposed team and the signals behind it, for many tickets at once."""
        idx, sims = self.neighbours(X)
        labels = self.index_labels[idx]
        pred = labels[:, 0]
        agreement = (labels == pred[:, None]).mean(axis=1)
        spread = sims[:, 0] - sims[:, -1]
        features = confidence_features(sims[:, 0], agreement, spread)
        z = features @ self.conf_coef + self.conf_intercept
        confidence = 1.0 / (1.0 + np.exp(-z))
        familiar = sims[:, 0] >= self.novelty_gate
        return {
            "pred": pred,
            "confidence": confidence,
            # What the answer/hand-over decision is based on: zero for unfamiliar tickets.
            "score": np.where(familiar, confidence, 0.0),
            "familiar": familiar,
            "top_sim": sims[:, 0],
            "agreement": agreement,
            "neighbour_idx": idx,
            "neighbour_sims": sims,
        }

    def decide(self, X: np.ndarray, threshold: float | None = None) -> list[dict]:
        """One decision per ticket, with the evidence a person would want to see."""
        tau = self.threshold if threshold is None else threshold
        p = self.propose(X)
        out = []
        for i in range(len(X)):
            idx, sims = p["neighbour_idx"][i], p["neighbour_sims"][i]
            votes: dict[str, int] = {}
            for j in idx:
                team = self.classes[self.index_labels[j]]
                votes[team] = votes.get(team, 0) + 1
            out.append(
                {
                    "team": self.classes[p["pred"][i]],
                    "confidence": float(p["confidence"][i]),
                    "handover": bool(p["score"][i] < tau),
                    "reason": (
                        "no similar past ticket"
                        if not p["familiar"][i]
                        else "confidence below threshold"
                        if p["score"][i] < tau
                        else "similar past tickets agree"
                    ),
                    "votes": [{"team": t, "neighbours": n} for t, n in sorted(votes.items(), key=lambda kv: -kv[1])],
                    "precedent": {
                        "text": self.index_texts[idx[0]],
                        "team": self.classes[self.index_labels[idx[0]]],
                        "similarity": float(sims[0]),
                    },
                }
            )
        return out

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path = ARTIFACT_PATH) -> Router:
        return joblib.load(path)

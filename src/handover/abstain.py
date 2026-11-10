"""When to answer and when to hand over to a person.

The business promise is: "among the tickets the system routes on its own,
at most `alpha` are wrong." This module picks the confidence threshold that
keeps that promise, with a statistical guarantee, and measures the
trade-off between automation rate (coverage) and error rate (risk).

Method: selective classification with guaranteed risk (Geifman and
El-Yaniv, 2017), using fixed-sequence testing (Angelopoulos et al.,
"Learn then Test", 2021) so that trying many thresholds does not weaken
the guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta


def risk_upper_bound(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper limit for an error rate.

    With probability at least 1 - delta, the true error rate is below this
    value, given `errors` mistakes seen in `n` independent tickets.
    """
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


@dataclass
class ThresholdChoice:
    threshold: float
    target_risk: float  # alpha: the promised error rate among answered tickets
    delta: float  # the promise may fail with at most this probability
    coverage: float  # share of calibration tickets answered
    observed_risk: float  # error rate among them on calibration data
    risk_bound: float  # the guaranteed upper limit
    answered: int

    def to_dict(self) -> dict:
        return asdict(self)


def choose_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
    delta: float = 0.05,
    steps: int = 200,
) -> ThresholdChoice:
    """Lowest confidence threshold whose guaranteed error rate stays under the target.

    Thresholds are tested from strict to lenient. The search stops at the
    first one that fails, which is what keeps the guarantee valid without
    a correction for multiple tests.

    `conf` and `correct` must come from data that was not used to train or
    calibrate the model.
    """
    candidates = np.quantile(conf, np.linspace(1.0, 0.0, steps + 1))
    best = ThresholdChoice(
        threshold=float("inf"),
        target_risk=target_risk,
        delta=delta,
        coverage=0.0,
        observed_risk=0.0,
        risk_bound=1.0,
        answered=0,
    )
    for tau in candidates:
        answered = conf >= tau
        n = int(answered.sum())
        errors = int((~correct[answered]).sum())
        bound = risk_upper_bound(errors, n, delta)
        if bound > target_risk:
            # Small strict sets can fail only because n is tiny; keep going
            # until something has passed, then stop at the first failure.
            if best.answered > 0:
                break
            continue
        best = ThresholdChoice(
            threshold=float(tau),
            target_risk=target_risk,
            delta=delta,
            coverage=n / len(conf),
            observed_risk=errors / n,
            risk_bound=bound,
            answered=n,
        )
    return best


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among answered tickets as more and more tickets are answered.

    Tickets are answered in order of confidence, most confident first.
    Returns (coverage, risk), both of length n.
    """
    order = np.argsort(-conf)
    wrong = (~correct[order]).astype(float)
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), np.cumsum(wrong) / n


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage_curve(conf, correct)
    return float(risk.mean())


def evaluate_threshold(conf: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """Coverage and error rate at a fixed threshold, on any dataset."""
    answered = conf >= threshold
    n = int(answered.sum())
    return {
        "coverage": n / len(conf),
        "answered": n,
        "risk": float((~correct[answered]).mean()) if n else 0.0,
        "accuracy_if_always_answering": float(correct.mean()),
    }

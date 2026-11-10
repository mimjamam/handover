import numpy as np

from handover.abstain import (
    aurc,
    choose_threshold,
    evaluate_threshold,
    risk_coverage_curve,
    risk_upper_bound,
)


def simulate(n: int, seed: int):
    """Tickets whose chance of being right equals the stated confidence."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.3, 1.0, n)
    correct = rng.uniform(0, 1, n) < conf
    return conf, correct


def test_bound_is_above_observed_rate():
    assert risk_upper_bound(5, 100, 0.05) > 0.05
    assert risk_upper_bound(0, 0, 0.05) == 1.0
    assert risk_upper_bound(10, 10, 0.05) == 1.0


def test_bound_tightens_with_more_data():
    assert risk_upper_bound(50, 1000, 0.05) < risk_upper_bound(5, 100, 0.05)


def test_threshold_keeps_the_promise_on_new_data():
    """The guarantee: on fresh data the error rate stays under the target.

    It may fail with probability delta, so across 40 repeats a few misses
    are allowed, but not many.
    """
    target, misses = 0.10, 0
    for seed in range(40):
        conf, correct = simulate(3000, seed)
        choice = choose_threshold(conf, correct, target_risk=target, delta=0.05)
        new_conf, new_correct = simulate(20000, 1000 + seed)
        result = evaluate_threshold(new_conf, new_correct, choice.threshold)
        misses += result["risk"] > target
    assert misses <= 6


def test_stricter_target_answers_fewer_tickets():
    conf, correct = simulate(5000, 0)
    loose = choose_threshold(conf, correct, target_risk=0.20)
    strict = choose_threshold(conf, correct, target_risk=0.08)
    assert strict.coverage < loose.coverage
    assert strict.threshold > loose.threshold


def test_impossible_target_answers_nothing():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 500)
    correct = np.zeros(500, dtype=bool)  # always wrong
    choice = choose_threshold(conf, correct, target_risk=0.05)
    assert choice.answered == 0
    assert evaluate_threshold(conf, correct, choice.threshold)["coverage"] == 0.0


def test_risk_coverage_curve_ends_at_overall_error():
    conf, correct = simulate(2000, 1)
    coverage, risk = risk_coverage_curve(conf, correct)
    assert coverage[-1] == 1.0
    assert abs(risk[-1] - (1 - correct.mean())) < 1e-9


def test_useful_confidence_beats_random_confidence():
    conf, correct = simulate(5000, 2)
    shuffled = np.random.default_rng(3).permutation(conf)
    assert aurc(conf, correct) < aurc(shuffled, correct)

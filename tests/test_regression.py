"""Fails the build if the saved model is worse than the recorded numbers.

The floors sit slightly below the values in reports/metrics.json, so a
real drop is caught and ordinary noise is not.
"""

import json
from pathlib import Path

import pytest

from handover.model import Router

METRICS = json.loads(Path("reports/metrics.json").read_text())
FLOORS = json.loads(Path("tests/floors.json").read_text())


@pytest.fixture(scope="module")
def router():
    return Router.load()


def test_model_file_matches_the_report(router):
    assert router.metadata["test"]["accuracy"] == pytest.approx(METRICS["test"]["accuracy"])
    assert router.classes == METRICS["classes"]


def test_accuracy_has_not_dropped():
    assert METRICS["test"]["accuracy"] >= FLOORS["accuracy"]
    assert METRICS["test"]["accuracy_de"] >= FLOORS["accuracy_de"]
    assert METRICS["test"]["macro_f1"] >= FLOORS["macro_f1"]


def test_confidence_is_calibrated():
    assert METRICS["test"]["ece_confidence"] <= FLOORS["max_ece"]
    assert METRICS["test"]["ece_confidence"] < METRICS["test"]["ece_raw_similarity"]


def test_confidence_is_informative():
    """Answering the most confident tickets first must beat answering at random."""
    assert METRICS["test"]["aurc"] < METRICS["test"]["aurc_if_confidence_were_random"]


def test_every_promise_is_kept_on_test_data():
    usable = [p for p in METRICS["operating_points"] if p["threshold"] is not None]
    assert usable, "no setting can keep its promise"
    for point in usable:
        assert point["promise_kept_on_test"], f"target {point['target_risk']} broken on test"


def test_default_setting_automates_enough():
    default = next(p for p in METRICS["operating_points"] if p["target_risk"] == METRICS["default_target_risk"])
    assert default["test"]["coverage"] >= FLOORS["default_coverage"]

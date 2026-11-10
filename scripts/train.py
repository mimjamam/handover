"""Build the router, choose the abstention thresholds, evaluate once on test.

Run:  uv run python scripts/train.py

Which split is used for what:
  train        the memory: past tickets the router looks up
  validation   fit the confidence model
  calibration  choose the abstention thresholds (untouched until then)
  test         final numbers
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from handover.abstain import aurc, choose_threshold, evaluate_threshold, risk_coverage_curve
from handover.data import DATASET_ID, load
from handover.embed import MODEL_NAME, embed
from handover.model import (
    K,
    Router,
    confidence_features,
    expected_calibration_error,
    fit_confidence_model,
)

EMB_PATH = Path("data/embeddings.npy")
REPORTS = Path("reports")
TARGET_RISKS = [0.01, 0.02, 0.05, 0.10]
DEFAULT_TARGET = 0.05
DELTA = 0.05
# Similarity bands used to show how results depend on having seen a similar ticket before.
BANDS = [
    (0.0, 0.95, "no similar past ticket"),
    (0.95, 0.97, "loosely similar"),
    (0.97, 0.985, "similar"),
    (0.985, 1.01, "near copy"),
]


def get_embeddings(texts: list[str]) -> np.ndarray:
    if EMB_PATH.exists():
        X = np.load(EMB_PATH)
        if len(X) == len(texts):
            return X
    X = embed(texts, progress=True)
    np.save(EMB_PATH, X)
    return X


def pick_examples(df_test, proposal, correct, threshold) -> list[dict]:
    """A few real test tickets for the demo page: some routed, some handed over."""
    rng = np.random.default_rng(7)
    length_ok = df_test["text"].str.len().between(120, 600).values
    lang = df_test["language"].values
    conf = proposal["score"]
    wanted = [
        ("de", "routed", (conf >= threshold) & correct),
        ("en", "routed", (conf >= threshold) & correct),
        ("de", "handed over", conf < threshold * 0.6),
        ("en", "handed over", conf < threshold * 0.6),
        ("de", "routed", (conf >= threshold) & correct),
        ("en", "handed over", (conf < threshold) & (conf > threshold * 0.6)),
    ]
    examples, used = [], set()
    for code, kind, mask in wanted:
        pool = [i for i in np.flatnonzero(mask & length_ok & (lang == code)) if i not in used]
        if not pool:
            continue
        i = int(rng.choice(pool))
        used.add(i)
        text = df_test["text"].iloc[i]
        title = text.split("\n")[0][:60]
        examples.append(
            {
                "label": f"{code.upper()} · {title}",
                "text": text,
                "expected": kind,
                "true_team": df_test["label"].iloc[i],
            }
        )
    return examples


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()
    X = get_embeddings(df["text"].tolist())
    idx = {s: np.flatnonzero(df["split"].values == s) for s in ("train", "validation", "calibration", "test")}
    classes = sorted(df["label"].unique())
    class_index = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(class_index).values
    lang = df["language"].values

    # 1. The memory: training tickets, stored at half precision to keep the file small.
    router = Router(
        classes=classes,
        index=X[idx["train"]].astype(np.float16),
        index_labels=y[idx["train"]].astype(np.int16),
        index_texts=df["text"].iloc[idx["train"]].tolist(),
    )

    def run(split: str):
        p = router.propose(X[idx[split]])
        return p, p["pred"] == y[idx[split]]

    # 2. Confidence model on validation.
    val, val_correct = run("validation")
    conf_model = fit_confidence_model(
        confidence_features(
            val["top_sim"], val["agreement"], val["neighbour_sims"][:, 0] - val["neighbour_sims"][:, -1]
        ),
        val_correct,
    )
    router.conf_coef = conf_model.coef_[0].astype(np.float64)
    router.conf_intercept = float(conf_model.intercept_[0])

    # Novelty gate: the similarity below which the closest past ticket is
    # wrong more often than right (measured on validation).
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val["top_sim"], val_correct)
    grid = np.linspace(0.80, 1.0, 2001)
    router.novelty_gate = float(grid[np.argmax(iso.predict(grid) >= 0.5)])

    # 3. Thresholds on calibration, checked on test.
    cal, cal_correct = run("calibration")
    test, test_correct = run("test")
    test_lang = lang[idx["test"]]
    operating_points, thresholds = [], {}
    for target in TARGET_RISKS:
        choice = choose_threshold(cal["score"], cal_correct, target_risk=target, delta=DELTA)
        on_test = evaluate_threshold(test["score"], test_correct, choice.threshold)
        point = {
            "target_risk": target,
            "threshold": choice.threshold if np.isfinite(choice.threshold) else None,
            "calibration": {"coverage": choice.coverage, "risk": choice.observed_risk, "risk_bound": choice.risk_bound},
            "test": {"coverage": on_test["coverage"], "risk": on_test["risk"], "answered": on_test["answered"]},
            "promise_kept_on_test": bool(on_test["risk"] <= target),
        }
        for code in ("de", "en"):
            m = test_lang == code
            r = evaluate_threshold(test["score"][m], test_correct[m], choice.threshold)
            point[f"test_{code}"] = {"coverage": r["coverage"], "risk": r["risk"]}
        operating_points.append(point)
        thresholds[f"{target:.2f}"] = choice.threshold
        if target == DEFAULT_TARGET:
            router.threshold = choice.threshold

    # 4. How results depend on having seen a similar ticket before.
    bands = []
    for lo, hi, name in BANDS:
        m = (test["top_sim"] >= lo) & (test["top_sim"] < hi)
        bands.append(
            {
                "band": name,
                "similarity_from": lo,
                "similarity_to": min(hi, 1.0),
                "share_of_tickets": float(m.mean()),
                "accuracy": float(test_correct[m].mean()) if m.any() else None,
                "share_routed_at_default": float((test["score"][m] >= router.threshold).mean()) if m.any() else None,
            }
        )

    # 5. Two ordinary classifiers for comparison (they always answer).
    y_tr, y_te = y[idx["train"]], y[idx["test"]]
    emb_lr = LogisticRegression(C=1.0, max_iter=3000).fit(X[idx["train"]], y_tr)
    emb_lr_pred = emb_lr.predict(X[idx["test"]])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    tfidf_lr = LogisticRegression(C=5, max_iter=2000).fit(vec.fit_transform(df["text"].iloc[idx["train"]]), y_tr)
    tfidf_pred = tfidf_lr.predict(vec.transform(df["text"].iloc[idx["test"]]))
    novel = test["top_sim"] < 0.95

    # 6. Final numbers on test.
    rng = np.random.default_rng(0)
    de, en = test_lang == "de", test_lang == "en"
    metrics = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": DATASET_ID,
        "embedding_model": MODEL_NAME,
        "neighbours": K,
        "novelty_gate": router.novelty_gate,
        "share_of_test_below_gate": float((~test["familiar"]).mean()),
        "rows": {s: len(i) for s, i in idx.items()},
        "classes": classes,
        "confidence_model": {"coef": router.conf_coef.tolist(), "intercept": router.conf_intercept},
        "test": {
            "accuracy": float(test_correct.mean()),
            "accuracy_de": float(test_correct[de].mean()),
            "accuracy_en": float(test_correct[en].mean()),
            "macro_f1": f1_score(y_te, test["pred"], average="macro"),
            "majority_class_accuracy": float((y_te == np.bincount(y_tr).argmax()).mean()),
            "ece_raw_similarity": expected_calibration_error(test["top_sim"], test_correct),
            "ece_confidence": expected_calibration_error(test["confidence"], test_correct),
            "aurc": aurc(test["score"], test_correct),
            "aurc_if_confidence_were_random": aurc(rng.permutation(test["score"]), test_correct),
        },
        "baselines_test": {
            "embedding_logistic_regression": {
                "accuracy": accuracy_score(y_te, emb_lr_pred),
                "accuracy_on_novel": accuracy_score(y_te[novel], emb_lr_pred[novel]),
            },
            "tfidf_logistic_regression": {
                "accuracy": accuracy_score(y_te, tfidf_pred),
                "accuracy_on_novel": accuracy_score(y_te[novel], tfidf_pred[novel]),
            },
            "router_accuracy_on_novel": float(test_correct[novel].mean()),
            "majority_class_on_novel": float((y_te[novel] == np.bincount(y_tr).argmax()).mean()),
        },
        "similarity_bands": bands,
        "default_target_risk": DEFAULT_TARGET,
        "delta": DELTA,
        "default_threshold": router.threshold if np.isfinite(router.threshold) else None,
        "operating_points": operating_points,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 7. Figure: the trade-off between automation and errors.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for code, name, colour in (
        ("all", "All tickets", "#1f2937"),
        ("en", "English", "#2563eb"),
        ("de", "German", "#d97706"),
    ):
        m = np.ones(len(test_correct), bool) if code == "all" else test_lang == code
        cov, risk = risk_coverage_curve(test["score"][m], test_correct[m])
        ax.plot(cov[20:] * 100, risk[20:] * 100, label=name, color=colour, linewidth=2 if code == "all" else 1.3)
    for p in operating_points:
        if p["threshold"] is not None:
            ax.scatter(p["test"]["coverage"] * 100, p["test"]["risk"] * 100, color="#1f2937", zorder=3, s=22)
            ax.annotate(
                f"limit {p['target_risk']:.0%}",
                (p["test"]["coverage"] * 100, p["test"]["risk"] * 100),
                textcoords="offset points",
                xytext=(-8, 8),
                fontsize=8,
                ha="right",
            )
    ax.set_xlabel("Tickets routed automatically (%)")
    ax.set_ylabel("Wrong among those (%)")
    ax.set_title("More automation costs more errors (test set)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "risk_coverage.png", dpi=160)

    # 8. Save the model file with a record of how it was built.
    router.metadata = {
        k: metrics[k]
        for k in (
            "created",
            "dataset",
            "embedding_model",
            "neighbours",
            "rows",
            "test",
            "similarity_bands",
            "default_target_risk",
            "delta",
            "operating_points",
        )
    }
    router.metadata["thresholds"] = thresholds
    router.metadata["examples"] = pick_examples(df.iloc[idx["test"]], test, test_correct, router.threshold)
    router.save()

    print(json.dumps(metrics["test"], indent=2))
    print(json.dumps(metrics["baselines_test"], indent=2))
    for b in bands:
        print(
            f"{b['band']:26s} share {b['share_of_tickets']:.1%}  accuracy {b['accuracy']:.1%}  routed {b['share_routed_at_default']:.1%}"
        )
    for p in operating_points:
        print(
            f"limit {p['target_risk']:.0%}: threshold {p['threshold']}  test coverage {p['test']['coverage']:.1%}  risk {p['test']['risk']:.1%}  "
            f"de {p['test_de']['coverage']:.1%}/{p['test_de']['risk']:.1%}  en {p['test_en']['coverage']:.1%}/{p['test_en']['risk']:.1%}  kept={p['promise_kept_on_test']}"
        )


if __name__ == "__main__":
    main()

"""Evaluate the final model on the held-out test patients and create figures."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src import config
from src.data import load_dataset, split_by_patient


def capacity_table(y_true, y_prob, levels) -> pd.DataFrame:
    """Precision/recall if we contact only the top-k% highest-risk patients.

    This matches how a hospital would use the model: a follow-up program has
    limited capacity, so we rank patients by risk instead of using a 0.5 cut.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob), kind="stable")
    base_rate = y_true.mean()
    rows = []
    for level in levels:
        n_flagged = int(round(level * len(y_true)))
        flagged = np.zeros(len(y_true), dtype=bool)
        flagged[order[:n_flagged]] = True
        true_positives = (flagged & (y_true == 1)).sum()
        precision = true_positives / n_flagged
        rows.append({
            "contacted_share": level,
            "precision": precision,
            "recall": true_positives / y_true.sum(),
            "lift": precision / base_rate,
        })
    return pd.DataFrame(rows)


def plot_roc_pr(y_true, y_prob, path=None):
    """ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name="Model")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_title("ROC curve")
    axes[0].legend()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name="Model")
    axes[1].axhline(np.mean(y_true), color="k", linestyle="--", label="Random (base rate)")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend()

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_calibration(y_true, probabilities: dict, path=None):
    """Reliability curve for one or more sets of predicted probabilities."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, y_prob in probabilities.items():
        CalibrationDisplay.from_predictions(
            y_true, y_prob, n_bins=10, strategy="quantile", ax=ax, name=name
        )
    ax.set_title("Calibration (reliability) curve")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_permutation_importance(model, X, y, path=None, top_n: int = 15):
    """Importance of raw input columns: drop in PR-AUC when a column is shuffled."""
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=5,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.Series(result.importances_mean, index=X.columns).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Mean decrease in PR-AUC")
    ax.set_title(f"Top {top_n} features (permutation importance)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Same split as in training (same seed), so test patients were never seen
    X, y, groups = load_dataset()
    _, X_test, _, y_test, _ = split_by_patient(X, y, groups, config.TEST_SIZE)

    calibrated = joblib.load(config.MODELS_DIR / "model_calibrated.joblib")
    uncalibrated = joblib.load(config.MODELS_DIR / "model_uncalibrated.joblib")
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    p_uncal = uncalibrated.predict_proba(X_test)[:, 1]

    metrics = {
        "n_test_encounters": int(len(y_test)),
        "base_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, p_cal)),
        "pr_auc": float(average_precision_score(y_test, p_cal)),
        "brier_uncalibrated": float(brier_score_loss(y_test, p_uncal)),
        "brier_calibrated": float(brier_score_loss(y_test, p_cal)),
    }
    capacity = capacity_table(y_test, p_cal, config.CAPACITY_LEVELS)

    with open(config.REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    capacity.to_csv(config.REPORTS_DIR / "capacity_table.csv", index=False)

    plot_roc_pr(y_test, p_cal, config.FIGURES_DIR / "roc_pr_curves.png")
    plot_calibration(
        y_test,
        {"Uncalibrated": p_uncal, "Calibrated (isotonic)": p_cal},
        config.FIGURES_DIR / "calibration_curve.png",
    )
    # A sample keeps permutation importance fast
    sample = X_test.sample(n=min(5000, len(X_test)), random_state=config.RANDOM_STATE)
    plot_permutation_importance(
        calibrated, sample, y_test.loc[sample.index],
        config.FIGURES_DIR / "permutation_importance.png",
    )

    plt.close("all")
    print(json.dumps(metrics, indent=2))
    print(capacity.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

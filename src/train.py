"""Compare models with patient-grouped CV, then train and calibrate the best one."""

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import config
from src.data import load_dataset, split_by_patient
from src.features import build_preprocessor, engineer_features

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
}


def build_models() -> dict:
    """Candidate models. Each pipeline takes raw rows as input.

    We do not use class weights or resampling: they distort predicted
    probabilities, and here we need reliable risk scores.
    """
    return {
        "logistic_regression": Pipeline([
            ("features", FunctionTransformer(engineer_features)),
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, C=0.5)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("features", FunctionTransformer(engineer_features)),
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False,
                random_state=config.RANDOM_STATE,
            )),
        ]),
    }


def compare_models(models: dict, X, y, groups) -> pd.DataFrame:
    """Run grouped, stratified CV and return mean/std of each metric."""
    cv = StratifiedGroupKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    rows = []
    for name, model in models.items():
        print(f"Cross-validating: {name}")
        scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            if metric == "brier":
                values = -values  # sklearn returns the negative Brier score
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_dataset()
    X_train, _, y_train, _, groups_train = split_by_patient(X, y, groups, config.TEST_SIZE)

    # 1) Model selection with cross-validation (the test set is not touched)
    models = build_models()
    cv_results = compare_models(models, X_train, y_train, groups_train)
    cv_results.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)
    print(cv_results.round(4).to_string(index=False))

    best_name = cv_results.iloc[0]["model"]
    print(f"Selected model (best PR-AUC): {best_name}")

    # 2) Fit on one part of the training patients, calibrate on the other part
    X_fit, X_calib, y_fit, y_calib, _ = split_by_patient(
        X_train, y_train, groups_train, config.CALIBRATION_SIZE
    )
    best_model = models[best_name].fit(X_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(best_model), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    # 3) Save artifacts
    joblib.dump(best_model, config.MODELS_DIR / "model_uncalibrated.joblib")
    joblib.dump(calibrated_model, config.MODELS_DIR / "model_calibrated.joblib")
    with open(config.MODELS_DIR / "model_info.json", "w") as f:
        json.dump({"selected_model": best_name}, f, indent=2)
    print(f"Models saved to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()

"""Unit tests for data preparation, feature engineering and the model pipeline."""

import numpy as np
import pytest

from src.data import prepare_dataset, split_by_patient
from src.evaluate import capacity_table
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, age_to_num, engineer_features, map_icd9
from src.train import build_models


@pytest.mark.parametrize("code, expected", [
    ("250.83", "Diabetes"),
    ("428", "Circulatory"),
    ("786", "Respiratory"),
    ("715", "Musculoskeletal"),
    ("996", "Injury"),
    ("V57", "Other"),
    (np.nan, "Missing"),
])
def test_map_icd9(code, expected):
    assert map_icd9(code) == expected


def test_age_to_num():
    assert age_to_num("[70-80)") == 75.0
    assert np.isnan(age_to_num(np.nan))


def test_prepare_dataset_removes_ids_and_excluded_discharges(raw_df):
    X, y, groups = prepare_dataset(raw_df)
    assert {"readmitted", "patient_nbr", "encounter_id"}.isdisjoint(X.columns)
    assert not X["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).any()
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(groups)


def test_engineer_features_columns(raw_df):
    X, _, _ = prepare_dataset(raw_df)
    features = engineer_features(X)
    assert list(features.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert (features["n_meds_active"] >= features["n_med_changes"]).all()


def test_split_has_no_patient_overlap(raw_df):
    X, y, groups = prepare_dataset(raw_df)
    X_train, X_test, _, _, _ = split_by_patient(X, y, groups, test_size=0.3)
    train_patients = set(groups.loc[X_train.index])
    test_patients = set(groups.loc[X_test.index])
    assert train_patients.isdisjoint(test_patients)


@pytest.mark.parametrize("model_name", ["logistic_regression", "hist_gradient_boosting"])
def test_models_fit_and_predict_probabilities(raw_df, model_name):
    X, y, _ = prepare_dataset(raw_df)
    model = build_models()[model_name].fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_capacity_table_perfect_ranking():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.linspace(1, 0, 10)  # positives get the highest scores
    table = capacity_table(y_true, y_prob, [0.2])
    assert table.loc[0, "precision"] == 1.0
    assert table.loc[0, "recall"] == 1.0

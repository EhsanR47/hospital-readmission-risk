"""Feature engineering and preprocessing.

Rule used in this project:
- Stateless, row-wise transformations (nothing learned from data) live in
  `engineer_features`.
- Anything that learns from data (imputation, scaling, encoding) lives inside
  the sklearn Pipeline, so it is fitted on training folds only (no leakage).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Numeric codes that are really categories (e.g. 1 = Emergency admission)
CODE_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "total_prior_visits", "n_med_changes", "n_meds_active",
]

CATEGORICAL_FEATURES = [
    "race", "gender", *CODE_COLUMNS, "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed", "insulin", "metformin",
    "diag_1_group", "diag_2_group", "diag_3_group",
]

# ICD-9 ranges (half-open intervals) mapped to clinical groups
ICD9_GROUPS = [
    ("Neoplasms", 140, 240),
    ("Circulatory", 390, 460),
    ("Respiratory", 460, 520),
    ("Digestive", 520, 580),
    ("Genitourinary", 580, 630),
    ("Musculoskeletal", 710, 740),
    ("Injury", 800, 1000),
]
# Symptom codes that belong to the same organ system
ICD9_SYMPTOM_CODES = {785: "Circulatory", 786: "Respiratory", 787: "Digestive", 788: "Genitourinary"}


def map_icd9(code) -> str:
    """Map a raw ICD-9 diagnosis code to a clinical group."""
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if value != value:  # float("nan") is not equal to itself
        return "Missing"

    if 250 <= value < 251:
        return "Diabetes"
    if int(value) in ICD9_SYMPTOM_CODES:
        return ICD9_SYMPTOM_CODES[int(value)]
    for group, low, high in ICD9_GROUPS:
        if low <= value < high:
            return group
    return "Other"


def age_to_num(age_bracket) -> float:
    """Convert an age bracket like '[70-80)' to its midpoint (75.0)."""
    if pd.isna(age_bracket):
        return float("nan")
    low, high = str(age_bracket).strip("[)").split("-")
    return (float(low) + float(high)) / 2


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from raw columns (stateless, row-wise)."""
    df = df.copy()

    df["age_num"] = df["age"].map(age_to_num)
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    medications = df[MEDICATION_COLUMNS]
    df["n_med_changes"] = medications.isin(["Up", "Down"]).sum(axis=1)
    df["n_meds_active"] = (medications != "No").sum(axis=1)

    for col in DIAGNOSIS_COLUMNS:
        df[f"{col}_group"] = df[col].map(map_icd9)
    for col in CODE_COLUMNS:
        df[col] = df[col].astype(str)

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Imputation + encoding. Scaling is only needed for linear models."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        # Rare categories (e.g. small medical specialties) are merged into one
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False
        )),
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

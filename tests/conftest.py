"""Shared test fixtures: a small synthetic dataset with the same raw schema."""

import numpy as np
import pandas as pd
import pytest

from src.features import MEDICATION_COLUMNS


def make_synthetic_raw(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Create fake rows with the raw column names and realistic value types."""
    rng = np.random.default_rng(seed)

    def choice(values, **kwargs):
        # object dtype keeps np.nan as a real missing value (not the string "nan")
        return rng.choice(np.array(values, dtype=object), n_rows, **kwargs)

    ages = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
    diag_codes = ["250.83", "428", "414", "786", "V57", "E888", "715", "996", np.nan]

    df = pd.DataFrame({
        "encounter_id": np.arange(n_rows),
        "patient_nbr": rng.integers(0, n_rows // 2, n_rows),
        "race": choice(["Caucasian", "AfricanAmerican", "Other", np.nan]),
        "gender": choice(["Female", "Male"]),
        "age": choice(ages),
        "admission_type_id": rng.integers(1, 8, n_rows),
        "discharge_disposition_id": rng.integers(1, 26, n_rows),
        "admission_source_id": rng.integers(1, 10, n_rows),
        "time_in_hospital": rng.integers(1, 15, n_rows),
        "medical_specialty": choice(["Cardiology", "InternalMedicine", np.nan]),
        "num_lab_procedures": rng.integers(1, 100, n_rows),
        "num_procedures": rng.integers(0, 7, n_rows),
        "num_medications": rng.integers(1, 60, n_rows),
        "number_outpatient": rng.poisson(0.4, n_rows),
        "number_emergency": rng.poisson(0.2, n_rows),
        "number_inpatient": rng.poisson(0.6, n_rows),
        "diag_1": choice(diag_codes),
        "diag_2": choice(diag_codes),
        "diag_3": choice(diag_codes),
        "number_diagnoses": rng.integers(1, 17, n_rows),
        "max_glu_serum": choice(["None", "Norm", ">200", ">300"]),
        "A1Cresult": choice(["None", "Norm", ">7", ">8"]),
        "change": choice(["No", "Ch"]),
        "diabetesMed": choice(["No", "Yes"]),
    })
    for col in MEDICATION_COLUMNS:
        df[col] = choice(["No", "Steady", "Up", "Down"], p=[0.7, 0.2, 0.05, 0.05])

    # Target loosely linked to prior inpatient visits, so models can learn something
    risk = 0.08 + 0.05 * df["number_inpatient"]
    df["readmitted"] = np.where(rng.random(n_rows) < risk, "<30",
                                choice([">30", "NO"]))
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_synthetic_raw()

"""Download, load and prepare the Diabetes 130-US Hospitals dataset."""

import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config

TARGET_COLUMN = "readmitted"
GROUP_COLUMN = "patient_nbr"
ID_COLUMN = "encounter_id"


def _find_file_in_zip(archive: zipfile.ZipFile, filename: str) -> bytes:
    """Return a file's content from a zip archive, also searching nested zips."""
    for name in archive.namelist():
        if name.endswith(filename):
            return archive.read(name)
    for name in archive.namelist():
        if name.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(archive.read(name)))
            try:
                return _find_file_in_zip(inner, filename)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"{filename} not found in the archive")


def download_data(force: bool = False) -> None:
    """Download the raw CSV from UCI if it is not already on disk."""
    if config.RAW_CSV.exists() and not force:
        print(f"Data already exists: {config.RAW_CSV.relative_to(config.ROOT_DIR)}")
        return

    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading dataset from UCI...")
        with urllib.request.urlopen(config.DATA_URL, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        content = _find_file_in_zip(archive, "diabetic_data.csv")
    except Exception as error:  # network error, changed URL or archive layout
        print(f"UCI download failed ({error}). Trying the GitHub mirror...")
        with urllib.request.urlopen(config.DATA_MIRROR_URL, timeout=120) as response:
            content = response.read()

    config.RAW_CSV.write_bytes(content)
    print(f"Saved to {config.RAW_CSV.relative_to(config.ROOT_DIR)}")


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV.

    In this dataset "?" means missing. The string "None" (e.g. in A1Cresult)
    means "test not performed", which is real information, so we disable
    pandas' default NA parsing to keep it as a category.
    """
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


def prepare_dataset(df: pd.DataFrame):
    """Filter invalid rows and return features (X), target (y) and patient groups.

    Target: 1 if the patient was readmitted within 30 days, else 0.
    """
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].isin(config.EXCLUDED_DISCHARGE_IDS)]
    df = df.reset_index(drop=True)

    y = (df[TARGET_COLUMN] == "<30").astype(int)
    groups = df[GROUP_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN, GROUP_COLUMN, ID_COLUMN])
    return X, y, groups


def load_dataset():
    """Load the raw file and return (X, y, groups)."""
    return prepare_dataset(load_raw())


def split_by_patient(X, y, groups, test_size: float, random_state: int = config.RANDOM_STATE):
    """Split so that each patient appears on only one side.

    Many patients have several encounters. A random row split would put the
    same patient in train and test, which leaks information and gives
    over-optimistic scores.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx],
    )


if __name__ == "__main__":
    download_data()
    X, y, groups = load_dataset()
    print(f"Encounters: {len(X):,} | Patients: {groups.nunique():,}")
    print(f"Positive rate (readmitted <30 days): {y.mean():.3f}")

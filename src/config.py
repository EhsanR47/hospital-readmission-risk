"""Central configuration: paths, data source and experiment settings."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT_DIR / "data" / "raw" / "diabetic_data.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Data source (UCI Machine Learning Repository, dataset id 296) ---
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
# Fallback: public GitHub copy of the same CSV, used if the UCI download fails
DATA_MIRROR_URL = (
    "https://raw.githubusercontent.com/markaljm/"
    "Diabetes-130-US-hospitals-for-years-1999-2008-Data-Set/main/"
    "dataset_diabetes/diabetic_data.csv"
)

# --- Experiment settings ---
RANDOM_STATE = 42
TEST_SIZE = 0.2          # share of patients kept for the final test set
CALIBRATION_SIZE = 0.2   # share of training patients used for calibration
CV_FOLDS = 5

# Discharge codes for death or hospice care.
# These patients cannot be readmitted, so they would add noisy negatives.
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# Share of patients a follow-up program can contact (used in evaluation).
CAPACITY_LEVELS = [0.10, 0.20, 0.30]

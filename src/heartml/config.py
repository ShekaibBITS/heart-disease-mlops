"""heartml.config

Notes (what this module does)
- Centralizes constants used across ingestion, preprocessing, training, and evaluation.
- Keeps URLs, column names, feature groups, and key hyperparameters in one place for reproducibility.
"""

# Import Path for OS-independent file path handling
from pathlib import Path  # Standard library utility for paths

# Define the project root as the directory that contains this file's grandparent (repo/src/heartml)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Resolve absolute path for reliability

# -----------------------------
# Dataset configuration
# -----------------------------

# Define the UCI URL used in the notebook (processed Cleveland dataset)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)  # Public dataset endpoint

# Define the feature/column names exactly as used in the notebook
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]  # 13 features + 1 target

# Define which features are treated as numerical (scaled)
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]  # As per notebook

# Define which features are treated as categorical/binary (left unscaled)
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]  # Remaining inputs

# Define the column that represents the label
TARGET_COL = "target"  # Output label column name

# -----------------------------
# Train/test split configuration
# -----------------------------

# Define test split size (same as notebook: 20%)
TEST_SIZE = 0.20  # Fraction of samples used for testing

# Define random seed (same as notebook)
RANDOM_STATE = 42  # Seed for reproducibility

# -----------------------------
# Model training configuration
# -----------------------------

# Logistic Regression defaults used in the notebook
LR_MAX_ITER = 1000  # Ensure convergence
LR_SOLVER = "lbfgs"  # Standard solver for binary classification

# Random Forest defaults used in the notebook
RF_N_ESTIMATORS = 100  # Number of trees
RF_MAX_DEPTH = 10  # Max depth per tree
RF_MIN_SAMPLES_SPLIT = 5  # Minimum samples required to split an internal node
RF_MIN_SAMPLES_LEAF = 2  # Minimum samples required to be at a leaf node
RF_N_JOBS = -1  # Use all CPU cores

# Cross-validation configuration (same as notebook: 5 folds)
CV_FOLDS = 5  # Number of folds for StratifiedKFold CV

# Grid-search hyperparameter grid (mirrors notebook intent; kept small for speed)
RF_PARAM_GRID = {
    "n_estimators": [100, 200],  # Candidate number of trees
    "max_depth": [8, 10, 12],  # Candidate maximum depths
    "min_samples_split": [2, 5, 10],  # Candidate split thresholds
    "min_samples_leaf": [1, 2, 4],  # Candidate leaf thresholds
}  # Parameter grid for GridSearchCV

# -----------------------------
# Paths for artifacts and outputs
# -----------------------------

# Define where to store raw downloaded data
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.csv"  # Stored as CSV for convenience

# Define where to store cleaned/processed data
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"  # Cleaned dataset

# Define where to store models
MODEL_DIR = PROJECT_ROOT / "models"  # Folder containing serialized models

# Define filenames for serialized artifacts
SCALER_PATH = MODEL_DIR / "scaler.joblib"  # StandardScaler artifact
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"  # Baseline model artifact
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"  # Baseline RF artifact
RF_TUNED_MODEL_PATH = MODEL_DIR / "random_forest_tuned_model.joblib"  # Tuned RF artifact

# Define where to store plots
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"  # Folder for images

# Define where to store metrics and tables
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"  # Folder for metrics
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"  # Model comparison table
METRICS_JSON_PATH = METRICS_DIR / "metrics.json"  # Full metrics dump as JSON

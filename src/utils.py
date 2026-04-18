"""
utils.py — Configuration constants and helpers for the Green-Tech AI Design Assistant.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "ENB2012_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "energy_surrogate.keras")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")

# ─── Dataset column mapping ─────────────────────────────────────────────────
FEATURE_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGET_COLUMNS = ["Y1", "Y2"]

FEATURE_NAMES = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution",
]

TARGET_NAMES = ["Heating Load", "Cooling Load"]

# ─── Feature bounds (from dataset, used by GA gene-space) ───────────────────
FEATURE_BOUNDS = {
    "X1": (0.62, 0.98),    # Relative Compactness
    "X2": (514.5, 808.5),  # Surface Area (m²)
    "X3": (245.0, 416.5),  # Wall Area (m²)
    "X4": (110.25, 220.5), # Roof Area (m²)
    "X5": (3.5, 7.0),      # Overall Height (m)
    "X6": (2, 5),           # Orientation (integer 2-5)
    "X7": (0.0, 0.40),     # Glazing Area (ratio)
    "X8": (0, 5),           # Glazing Area Distribution (integer 0-5)
}

# ─── Neural Network hyper-parameters ────────────────────────────────────────
NN_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout_rates": [0.3, 0.2, 0.0],
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 500,
    "patience": 20,
    "validation_split": 0.2,
}

# ─── Genetic Algorithm hyper-parameters ──────────────────────────────────────
GA_CONFIG = {
    "num_generations": 200,
    "num_parents_mating": 10,
    "sol_per_pop": 50,
    "mutation_percent_genes": 15,
    "crossover_type": "single_point",
    "mutation_type": "random",
    "parent_selection_type": "tournament",
    "keep_elitism": 2,
}

# ─── SHAP config ─────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 100


def ensure_dirs():
    """Create project directories if they don't exist."""
    for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

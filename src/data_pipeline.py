"""
Advanced data loading, cleaning, and feature engineering pipeline.
Includes rich interaction features, log-transforms, and cyclical encodings
to maximise classifier accuracy.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, AUDIO_FEATURES, TARGET_COL, RANDOM_STATE

logger = logging.getLogger(__name__)


# ── Raw loading ───────────────────────────────────────────────────────────────
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and perform basic sanity checks on the raw CSV."""
    logger.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle nulls, and drop unusable rows."""
    df = df.drop_duplicates(subset=["track_id"])
    df = df.dropna(subset=AUDIO_FEATURES + [TARGET_COL])

    # Remove extreme-duration tracks (< 30 s or > 15 min)
    df = df[df["duration_ms"].between(30_000, 900_000)]

    # Clip audio features to Spotify specification range
    clip_map = {
        "danceability":     (0.0, 1.0),
        "energy":           (0.0, 1.0),
        "speechiness":      (0.0, 1.0),
        "acousticness":     (0.0, 1.0),
        "instrumentalness": (0.0, 1.0),
        "liveness":         (0.0, 1.0),
        "valence":          (0.0, 1.0),
    }
    for col, (lo, hi) in clip_map.items():
        df[col] = df[col].clip(lo, hi)

    logger.info(f"Clean shape: {df.shape}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional derived features that give classifiers extra signal.

    Strategies:
      • Interaction ratios / products
      • Log-transformations for heavy-tailed distributions
      • Cyclical encoding of musical key (12-tone circle)
      • Popularity buckets
    """
    df = df.copy()

    # ── Ratios & interactions ─────────────────────────────────────────────────
    df["energy_dance_ratio"]      = df["energy"] / (df["danceability"] + 1e-6)
    df["acoustic_energy_diff"]    = df["acousticness"] - df["energy"]
    df["speech_instrument_ratio"] = df["speechiness"] / (df["instrumentalness"] + 1e-6)
    df["valence_energy"]          = df["valence"] * df["energy"]
    df["valence_dance"]           = df["valence"] * df["danceability"]
    df["energy_loudness"]         = df["energy"] * (df["loudness"] + 60) / 60   # norm loudness
    df["acoustic_instrument"]     = df["acousticness"] * df["instrumentalness"]
    df["energy_valence_sum"]      = df["energy"] + df["valence"]

    # ── Log-transforms (instrument & speech are very skewed) ──────────────────
    df["log_instrumentalness"] = np.log1p(df["instrumentalness"])
    df["log_speechiness"]      = np.log1p(df["speechiness"])
    df["log_liveness"]         = np.log1p(df["liveness"])

    # ── Duration features ─────────────────────────────────────────────────────
    df["duration_min"]         = df["duration_ms"] / 60_000
    df["log_duration_ms"]      = np.log1p(df["duration_ms"])

    # ── Cyclical key encoding (circle of fifths, 12 tones) ───────────────────
    df["key_sin"] = np.sin(2 * np.pi * df["key"] / 12)
    df["key_cos"] = np.cos(2 * np.pi * df["key"] / 12)

    # ── Popularity buckets ────────────────────────────────────────────────────
    df["popularity_bin"] = pd.cut(
        df["track_popularity"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # ── Tempo features ────────────────────────────────────────────────────────
    df["tempo_energy"]   = df["tempo"] * df["energy"]
    df["tempo_dance"]    = df["tempo"] * df["danceability"]

    return df


# ── Feature catalogue ─────────────────────────────────────────────────────────
ENGINEERED_FEATURES = [
    "energy_dance_ratio", "acoustic_energy_diff", "speech_instrument_ratio",
    "valence_energy", "valence_dance", "energy_loudness", "acoustic_instrument",
    "energy_valence_sum",
    "log_instrumentalness", "log_speechiness", "log_liveness",
    "duration_min", "log_duration_ms",
    "key_sin", "key_cos",
    "popularity_bin",
    "tempo_energy", "tempo_dance",
]

ALL_FEATURES = AUDIO_FEATURES + ENGINEERED_FEATURES


# ── Feature matrix builder ────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    """Return X (numpy float32), y (numpy int), scaler, label_encoder."""
    df = feature_engineer(df)

    X = df[ALL_FEATURES].values.astype(np.float32)
    y_raw = df[TARGET_COL].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le


# ── SMOTE ─────────────────────────────────────────────────────────────────────
def apply_smote(X, y, random_state: int = RANDOM_STATE):
    """Balance classes via SMOTE (call only on training fold to avoid leakage)."""
    logger.info("Applying SMOTE …")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"Post-SMOTE shape: {X_res.shape}")
    return X_res, y_res


# ── Convenience wrapper ───────────────────────────────────────────────────────
def get_full_pipeline(path: str = DATA_PATH):
    """Load → clean → feature-engineer.  SMOTE applied per-fold inside trainer."""
    df = load_raw_data(path)
    df = clean_data(df)
    X, y, scaler, le = build_feature_matrix(df)
    return X, y, scaler, le, df

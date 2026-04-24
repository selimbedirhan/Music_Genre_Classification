"""
Data loading, cleaning, and feature engineering pipeline.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, AUDIO_FEATURES, TARGET_COL, RANDOM_STATE

logger = logging.getLogger(__name__)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and do basic sanity checks on the raw CSV."""
    logger.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle nulls, and drop unusable rows."""
    df = df.drop_duplicates(subset=["track_id"])
    df = df.dropna(subset=AUDIO_FEATURES + [TARGET_COL])

    # Remove extreme-duration outliers (< 30s or > 15min)
    df = df[df["duration_ms"].between(30_000, 900_000)]

    # Clip audio feature ranges to Spotify spec
    clip_map = {
        "danceability": (0, 1), "energy": (0, 1),
        "speechiness": (0, 1),  "acousticness": (0, 1),
        "instrumentalness": (0, 1), "liveness": (0, 1), "valence": (0, 1),
    }
    for col, (lo, hi) in clip_map.items():
        df[col] = df[col].clip(lo, hi)

    logger.info(f"Clean shape: {df.shape}")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features."""
    df = df.copy()
    df["energy_dance_ratio"]     = df["energy"] / (df["danceability"] + 1e-6)
    df["acoustic_energy_diff"]   = df["acousticness"] - df["energy"]
    df["speech_instrument_ratio"]= df["speechiness"] / (df["instrumentalness"] + 1e-6)
    df["duration_min"]           = df["duration_ms"] / 60_000
    df["valence_energy"]         = df["valence"] * df["energy"]
    return df


ENGINEERED_FEATURES = [
    "energy_dance_ratio", "acoustic_energy_diff",
    "speech_instrument_ratio", "duration_min", "valence_energy",
]

ALL_FEATURES = AUDIO_FEATURES + ENGINEERED_FEATURES


def build_feature_matrix(df: pd.DataFrame):
    """Return X (numpy array), y (numpy array), scaler, label_encoder."""
    df = feature_engineer(df)

    X = df[ALL_FEATURES].values.astype(np.float32)
    y_raw = df[TARGET_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le


def apply_smote(X, y, random_state: int = RANDOM_STATE):
    """Balance classes via SMOTE."""
    logger.info("Applying SMOTE …")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"Post-SMOTE shape: {X_res.shape}")
    return X_res, y_res


def get_full_pipeline(path: str = DATA_PATH):
    """Convenience wrapper: load → clean → features (no SMOTE here).
    SMOTE is applied per-fold inside the trainer to avoid leakage."""
    df = load_raw_data(path)
    df = clean_data(df)
    X, y, scaler, le = build_feature_matrix(df)
    return X, y, scaler, le, df

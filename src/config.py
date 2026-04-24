"""
Central configuration for the Music Genre Classification project.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "archive", "spotify_songs.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
TARGET_COL = "playlist_genre"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "key", "mode", "track_popularity",
]

GENRE_COLORS = {
    "edm":   "#00D4FF",
    "pop":   "#FF6B9D",
    "rap":   "#FFD700",
    "r&b":  "#C850C0",
    "latin": "#FF8C42",
    "rock":  "#7FFF00",
}

GENRE_EMOJIS = {
    "edm":   "⚡",
    "pop":   "🌟",
    "rap":   "🎤",
    "r&b":  "🎵",
    "latin": "💃",
    "rock":  "🎸",
}

# ── Random state ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Model artifact names ──────────────────────────────────────────────────────
BEST_MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODER_PATH     = os.path.join(MODELS_DIR, "label_encoder.joblib")
RESULTS_PATH     = os.path.join(MODELS_DIR, "benchmark_results.json")

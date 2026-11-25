"""Paths and constants shared across ingest, features, and training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GAME_LOG_DIR = RAW_DIR / "game_logs"
PBP_DIR = RAW_DIR / "pbp"
BOX_DIR = RAW_DIR / "box"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
EVAL_DIR = DATA_DIR / "eval"

# 2021-22 is only a shrinkage prior for pregame ratings. Training snapshots
# start in 2022-23. 2026-27 is in progress and is not used.
PRIOR_SEASON = "2021-22"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
GAME_LOG_SEASONS = [PRIOR_SEASON, *SEASONS]

# First day of the 2025-26 regular season window. Earlier games are training.
VALIDATION_SEASON_START = "2025-10-01"

PERIOD_SECONDS = 12 * 60
REGULATION_PERIODS = 4
SNAPSHOT_STRIDE_SECONDS = 60
# Elapsed times 0, 60, ..., 2820: 48 regulation snapshots, including the tip.
SNAPSHOT_COUNT = (REGULATION_PERIODS * PERIOD_SECONDS) // SNAPSHOT_STRIDE_SECONDS

ROLLING_GAMES = 20
SHRINKAGE_K = 10

REQUEST_SLEEP_SECONDS = 0.6
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60

# Support used when turning a negative-binomial into an explicit PMF.
REMAINING_PMF_MAX = 350
RELIABILITY_THRESHOLD = 220

NUMERIC_FEATURES = [
    "period",
    "seconds_left_in_period",
    "seconds_left_in_regulation",
    "current_total",
    "score_margin",
    "home_fouls_period",
    "away_fouls_period",
    "home_pace",
    "home_off_rating",
    "home_def_rating",
    "away_pace",
    "away_off_rating",
    "away_def_rating",
]

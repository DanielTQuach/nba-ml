"""Train the clock baseline and the remaining-points network."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from nba_ml.config import MODEL_DIR, NUMERIC_FEATURES, PROCESSED_DIR, VALIDATION_SEASON_START
from nba_ml.models.baseline import EmpiricalRemaining
from nba_ml.models.remaining import RemainingPointsNet, negative_log_likelihood

PLAYER_SLOTS = 10
MINUTES_SCALE = 48.0
FOULS_SCALE = 6.0


def assign_splits(snapshots: pd.DataFrame) -> pd.Series:
    """Train on seasons before 2025-26. Split that season in half by game date."""
    game_dates = snapshots.groupby("game_id")["game_date"].min()
    cutoff = pd.Timestamp(VALIDATION_SEASON_START)
    holdout = game_dates[game_dates >= cutoff]
    if holdout.empty:
        raise ValueError(f"no games on or after {VALIDATION_SEASON_START}")
    midpoint = holdout.median()
    labels = {}
    for game_id, game_date in game_dates.items():
        if game_date < cutoff:
            labels[game_id] = "train"
        elif game_date <= midpoint:
            labels[game_id] = "val"
        else:
            labels[game_id] = "test"
    assigned = snapshots["game_id"].map(labels)
    if (assigned == "test").sum() == 0 or (assigned == "val").sum() == 0:
        raise ValueError("validation season did not produce both a validation and a test half")
    return assigned


def _index_map(values: pd.Series) -> dict[int, int]:
    mapping = {0: 0}
    for offset, value in enumerate(sorted({int(item) for item in values if pd.notna(item)}), start=1):
        mapping[int(value)] = offset
    return mapping


def _encode(values: pd.Series, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(value), 0) if pd.notna(value) else 0 for value in values], dtype=np.int64)


def lineup_matrix(frame: pd.DataFrame, player_to_idx: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    players = np.zeros((len(frame), PLAYER_SLOTS), dtype=np.int64)
    minutes = np.zeros((len(frame), PLAYER_SLOTS), dtype=np.float32)
    fouls = np.zeros((len(frame), PLAYER_SLOTS), dtype=np.float32)
    for slot in range(5):
        for side, offset in (("home", 0), ("away", 5)):
            column = offset + slot
            players[:, column] = _encode(frame[f"{side}_p{slot + 1}"], player_to_idx)
            minutes[:, column] = frame[f"{side}_m{slot + 1}"].to_numpy(dtype=float) / MINUTES_SCALE
            fouls[:, column] = frame[f"{side}_f{slot + 1}"].to_numpy(dtype=float) / FOULS_SCALE
    return players, minutes.astype(np.float32), fouls.astype(np.float32)


class SnapshotDataset(Dataset):
    def __init__(
        self,
        numeric: np.ndarray,
        home: np.ndarray,
        away: np.ndarray,
        target: np.ndarray,
        players: np.ndarray | None = None,
        minutes: np.ndarray | None = None,
        fouls: np.ndarray | None = None,
    ) -> None:
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.home = torch.tensor(home, dtype=torch.long)
        self.away = torch.tensor(away, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.players = None if players is None else torch.tensor(players, dtype=torch.long)
        self.minutes = None if minutes is None else torch.tensor(minutes, dtype=torch.float32)
        self.fouls = None if fouls is None else torch.tensor(fouls, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "numeric": self.numeric[index],
            "home": self.home[index],
            "away": self.away[index],
            "target": self.target[index],
        }
        if self.players is not None:
            item["players"] = self.players[index]
            item["minutes"] = self.minutes[index]
            item["fouls"] = self.fouls[index]
        return item


def _batch_loss(model: RemainingPointsNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mu, alpha = model(
        batch["numeric"],
        batch["home"],
        batch["away"],
        batch.get("players"),
        batch.get("minutes"),
        batch.get("fouls"),
    )
    return negative_log_likelihood(mu, alpha, batch["target"])


def _epoch_loss(model: RemainingPointsNet, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        if training:
            optimizer.zero_grad()
        loss = _batch_loss(model, batch)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.item()) * len(batch["target"])
        count += len(batch["target"])
    return total / max(count, 1)


def load_training_frame(phase: int) -> pd.DataFrame:
    snapshots = pd.read_parquet(PROCESSED_DIR / "snapshots.parquet")
    snapshots["game_date"] = pd.to_datetime(snapshots["game_date"])
    if phase == 1:
        return snapshots
    lineups = pd.read_parquet(PROCESSED_DIR / "lineups.parquet")
    merged = snapshots.merge(lineups, on=["game_id", "period", "seconds_left_in_period"], how="inner")
    if merged.empty:
        raise ValueError("lineup features did not match any snapshots")
    return merged


def fit_baselines(train: pd.DataFrame, directory: Path) -> None:
    clock = EmpiricalRemaining(with_margin=False)
    clock.fit(train["seconds_left_in_regulation"].to_numpy(), train["remaining_points"].to_numpy())
    clock.save(directory / "baseline_clock.npz")
    margin = EmpiricalRemaining(with_margin=True)
    margin.fit(
        train["seconds_left_in_regulation"].to_numpy(),
        train["remaining_points"].to_numpy(),
        train["score_margin"].to_numpy(),
    )
    margin.save(directory / "baseline_margin.npz")


def train_model(frame: pd.DataFrame, phase: int, epochs: int = 30, patience: int = 5, batch_size: int = 2048) -> Path:
    torch.manual_seed(0)
    np.random.seed(0)
    frame = frame.copy()
    frame["split"] = assign_splits(frame)
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    use_lineups = phase == 2
    team_to_idx = _index_map(pd.concat([train["home_team_id"], train["away_team_id"]], ignore_index=True))
    mean = train[NUMERIC_FEATURES].to_numpy(dtype=float).mean(axis=0)
    std = train[NUMERIC_FEATURES].to_numpy(dtype=float).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    def numeric_matrix(part: pd.DataFrame) -> np.ndarray:
        values = part[NUMERIC_FEATURES].to_numpy(dtype=float)
        return ((values - mean) / std).astype(np.float32)

    player_to_idx: dict[int, int] = {0: 0}
    train_players = val_players = train_minutes = val_minutes = train_fouls = val_fouls = None
    if use_lineups:
        player_columns = [f"{side}_p{slot}" for side in ("home", "away") for slot in range(1, 6)]
        player_to_idx = _index_map(pd.concat([train[column] for column in player_columns], ignore_index=True))
        train_players, train_minutes, train_fouls = lineup_matrix(train, player_to_idx)
        val_players, val_minutes, val_fouls = lineup_matrix(val, player_to_idx)

    train_ds = SnapshotDataset(
        numeric_matrix(train),
        _encode(train["home_team_id"], team_to_idx),
        _encode(train["away_team_id"], team_to_idx),
        train["remaining_points"].to_numpy(dtype=float),
        train_players,
        train_minutes,
        train_fouls,
    )
    val_ds = SnapshotDataset(
        numeric_matrix(val),
        _encode(val["home_team_id"], team_to_idx),
        _encode(val["away_team_id"], team_to_idx),
        val["remaining_points"].to_numpy(dtype=float),
        val_players,
        val_minutes,
        val_fouls,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = RemainingPointsNet(
        n_teams=max(team_to_idx.values()) + 1,
        numeric_dim=len(NUMERIC_FEATURES),
        n_players=max(player_to_idx.values()) + 1,
        use_lineups=use_lineups,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    waits = 0
    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(model, train_loader, optimizer)
        with torch.no_grad():
            val_loss = _epoch_loss(model, val_loader, optimizer=None)
        print(f"epoch {epoch} train_nll={train_loss:.4f} val_nll={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            waits = 0
        else:
            waits += 1
            if waits >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if phase == 1:
        fit_baselines(train, MODEL_DIR)
    destination = MODEL_DIR / f"phase{phase}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "team_to_idx": team_to_idx,
            "player_to_idx": player_to_idx,
            "numeric_mean": mean.tolist(),
            "numeric_std": std.tolist(),
            "numeric_features": list(NUMERIC_FEATURES),
            "use_lineups": use_lineups,
            "n_teams": max(team_to_idx.values()) + 1,
            "n_players": max(player_to_idx.values()) + 1,
            "best_val_nll": best_loss,
        },
        destination,
    )
    print(f"wrote {destination} val_nll={best_loss:.4f}", flush=True)
    counts = frame["split"].value_counts().to_dict()
    print(f"rows {counts}", flush=True)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train remaining-points models.")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    frame = load_training_frame(args.phase)
    train_model(frame, phase=args.phase, epochs=args.epochs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise

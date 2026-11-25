"""On-court players, minutes, and fouls at each minute snapshot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from nba_ml.config import MAX_RETRIES, PERIOD_SECONDS, PROCESSED_DIR, RAW_DIR, REQUEST_SLEEP_SECONDS, REQUEST_TIMEOUT, SEASONS
from nba_ml.features.snapshots import _event_is_after, regulation_targets
from nba_ml.ingest.game_logs import load_games, normalize_game_id
from nba_ml.ingest.pbp import load_pbp, pbp_path

OT_SECONDS = 5 * 60
ROTATION_DIR = RAW_DIR / "rotation"


def _fetch_rotation(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import GameRotation

    last_error: Exception | None = None
    game_id = normalize_game_id(game_id)
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            endpoint = GameRotation(game_id=game_id, timeout=REQUEST_TIMEOUT)
            home = endpoint.home_team.get_data_frame()
            away = endpoint.away_team.get_data_frame()
            if home is None or away is None or home.empty or away.empty:
                raise RuntimeError("game rotation returned no stints")
            return normalize_rotation(home, away)
        except Exception as exc:  # noqa: BLE001 - network and API errors retry together
            last_error = exc
            time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise RuntimeError(f"game rotation for {game_id} failed after {MAX_RETRIES} tries") from last_error


def normalize_rotation(home: pd.DataFrame, away: pd.DataFrame) -> pd.DataFrame:
    """Stints from the league rotation feed. Times are tenths of a second from tip."""
    frames = []
    for raw in (home, away):
        frames.append(
            pd.DataFrame(
                {
                    "team_id": pd.to_numeric(raw["TEAM_ID"], errors="coerce").astype(int),
                    "person_id": pd.to_numeric(raw["PERSON_ID"], errors="coerce").astype(int),
                    "in_tenths": pd.to_numeric(raw["IN_TIME_REAL"], errors="coerce").astype(float),
                    "out_tenths": pd.to_numeric(raw["OUT_TIME_REAL"], errors="coerce").astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def rotation_path(game_id: str) -> Path:
    return ROTATION_DIR / f"{normalize_game_id(game_id)}.parquet"


def ingest_rotations(seasons: list[str] | None = None) -> dict[str, int]:
    seasons = seasons or list(SEASONS)
    games = load_games()
    selected = games[games["season"].isin(seasons)]
    ROTATION_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = cached = failed = 0
    total = len(selected)
    for index, game in enumerate(selected.itertuples(index=False), start=1):
        destination = rotation_path(game.game_id)
        if destination.exists() and destination.stat().st_size > 0:
            cached += 1
            continue
        try:
            _fetch_rotation(game.game_id).to_parquet(destination, index=False)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - one bad game should not stop the cache
            failed += 1
            print(f"failed rotation {game.game_id}: {exc}", flush=True)
        if index % 25 == 0 or index == total:
            print(
                f"rotation {index}/{total} downloaded={downloaded} cached={cached} failed={failed}",
                flush=True,
            )
    return {"downloaded": downloaded, "cached": cached, "failed": failed}


def starter_ids(box: pd.DataFrame, team_id: int) -> list[int]:
    team = box[box["team_id"] == int(team_id)]
    starters = team[team["position"].astype(str).str.len() > 0]
    return [int(person_id) for person_id in starters["person_id"].tolist()]


def _elapsed(period: int, seconds_left: float) -> float:
    period = int(period)
    if period <= 4:
        return (period - 1) * PERIOD_SECONDS + (PERIOD_SECONDS - float(seconds_left))
    return 4 * PERIOD_SECONDS + (period - 5) * OT_SECONDS + (OT_SECONDS - float(seconds_left))


def _is_personal_foul(action_type: object, sub_type: object) -> bool:
    if str(action_type).strip().lower() != "foul":
        return False
    return "technical" not in str(sub_type or "").lower()


def _apply_sub(on_court: set[int], person_out: object, person_in: object) -> None:
    if pd.notna(person_out):
        on_court.discard(int(person_out))
    if pd.notna(person_in):
        on_court.add(int(person_in))


def lineup_snapshots(
    events: pd.DataFrame,
    home_starters: list[int],
    away_starters: list[int],
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame | None:
    """Return minute rows, or None when the court is not 5 against 5 at every snapshot."""
    if len(home_starters) != 5 or len(set(home_starters)) != 5:
        return None
    if len(away_starters) != 5 or len(set(away_starters)) != 5:
        return None
    ordered = events.sort_values(["period", "seconds_left", "action_number"], ascending=[True, False, True])
    home_on = set(int(player_id) for player_id in home_starters)
    away_on = set(int(player_id) for player_id in away_starters)
    seconds_played: dict[int, float] = {player_id: 0.0 for player_id in home_on | away_on}
    fouls: dict[int, int] = {player_id: 0 for player_id in home_on | away_on}
    last_elapsed = 0.0
    rows = []
    event_index = 0
    records = ordered.to_dict("records")

    def apply(event: dict) -> None:
        nonlocal last_elapsed
        now = _elapsed(event["period"], event["seconds_left"])
        delta = max(0.0, now - last_elapsed)
        for player_id in home_on | away_on:
            seconds_played[player_id] = seconds_played.get(player_id, 0.0) + delta
        last_elapsed = now
        action = str(event["action_type"]).lower()
        team_id = event["team_id"]
        if action == "substitution" and pd.notna(team_id):
            side = home_on if int(team_id) == int(home_team_id) else away_on if int(team_id) == int(away_team_id) else None
            if side is not None:
                _apply_sub(side, event["person_out"], event["person_in"])
                for player_id in side:
                    seconds_played.setdefault(player_id, 0.0)
                    fouls.setdefault(player_id, 0)
        if _is_personal_foul(action, event.get("sub_type")) and pd.notna(event.get("person_id")):
            player_id = int(event["person_id"])
            fouls[player_id] = fouls.get(player_id, 0) + 1

    def capture(period: int, seconds_left: int) -> dict | None:
        if len(home_on) != 5 or len(away_on) != 5:
            return None
        home_ids = sorted(home_on)
        away_ids = sorted(away_on)
        row = {"period": period, "seconds_left_in_period": seconds_left}
        for slot, player_id in enumerate(home_ids, start=1):
            row[f"home_p{slot}"] = player_id
            row[f"home_m{slot}"] = seconds_played.get(player_id, 0.0) / 60.0
            row[f"home_f{slot}"] = fouls.get(player_id, 0)
        for slot, player_id in enumerate(away_ids, start=1):
            row[f"away_p{slot}"] = player_id
            row[f"away_m{slot}"] = seconds_played.get(player_id, 0.0) / 60.0
            row[f"away_f{slot}"] = fouls.get(player_id, 0)
        return row

    for period, seconds_left in regulation_targets():
        while event_index < len(records) and not _event_is_after(
            int(records[event_index]["period"]),
            float(records[event_index]["seconds_left"]),
            period,
            seconds_left,
        ):
            apply(records[event_index])
            event_index += 1
        row = capture(period, seconds_left)
        if row is None:
            return None
        rows.append(row)
    return pd.DataFrame(rows)


def _minutes_played(stints: pd.DataFrame, person_id: int, tenths: float) -> float:
    player = stints[stints["person_id"] == int(person_id)]
    overlap = (player["out_tenths"].clip(upper=tenths) - player["in_tenths"]).clip(lower=0.0)
    return float(overlap.sum()) / 600.0


def rotation_snapshots(
    stints: pd.DataFrame,
    events: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame | None:
    """On-court players from league stints. None when a minute is not 5 against 5.

    Play-by-play substitution text is not a consistent five-man ledger, so the
    on-court group and the minutes come from the game rotation feed. Personal
    fouls still come from the play-by-play up to that minute.
    """
    foul_events = []
    if not events.empty:
        ordered = events.sort_values(["period", "seconds_left", "action_number"], ascending=[True, False, True])
        for event in ordered.itertuples(index=False):
            if not _is_personal_foul(event.action_type, getattr(event, "sub_type", "")):
                continue
            person_id = getattr(event, "person_id", None)
            if person_id is None or pd.isna(person_id):
                continue
            foul_events.append((_elapsed(event.period, event.seconds_left), int(person_id)))
    foul_events.sort(key=lambda item: item[0])
    rows = []
    foul_index = 0
    foul_counts: dict[int, int] = {}
    for period, seconds_left in regulation_targets():
        elapsed = _elapsed(period, seconds_left)
        while foul_index < len(foul_events) and foul_events[foul_index][0] <= elapsed + 1e-6:
            player_id = foul_events[foul_index][1]
            foul_counts[player_id] = foul_counts.get(player_id, 0) + 1
            foul_index += 1
        tenths = elapsed * 10.0
        active = stints[(stints["in_tenths"] <= tenths) & (stints["out_tenths"] > tenths)]
        home_ids = sorted(
            int(player_id) for player_id in active.loc[active["team_id"] == int(home_team_id), "person_id"]
        )
        away_ids = sorted(
            int(player_id) for player_id in active.loc[active["team_id"] == int(away_team_id), "person_id"]
        )
        if len(home_ids) != 5 or len(set(home_ids)) != 5 or len(away_ids) != 5 or len(set(away_ids)) != 5:
            return None
        row: dict[str, float | int] = {"period": period, "seconds_left_in_period": seconds_left}
        for slot, player_id in enumerate(home_ids, start=1):
            row[f"home_p{slot}"] = player_id
            row[f"home_m{slot}"] = _minutes_played(stints, player_id, tenths)
            row[f"home_f{slot}"] = foul_counts.get(player_id, 0)
        for slot, player_id in enumerate(away_ids, start=1):
            row[f"away_p{slot}"] = player_id
            row[f"away_m{slot}"] = _minutes_played(stints, player_id, tenths)
            row[f"away_f{slot}"] = foul_counts.get(player_id, 0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    id_columns = [column for column in frame.columns if column == "period" or column.startswith(("home_p", "away_p", "home_f", "away_f"))]
    frame[id_columns] = frame[id_columns].astype(int)
    return frame


def build_lineups() -> pd.DataFrame:
    games = load_games()
    games = games[games["season"].isin(SEASONS)]
    frames = []
    invalid = missing = 0
    for game in games.itertuples(index=False):
        events_path = pbp_path(game.game_id)
        stint_path = rotation_path(game.game_id)
        if not events_path.exists() or not stint_path.exists():
            missing += 1
            continue
        built = rotation_snapshots(
            pd.read_parquet(stint_path),
            load_pbp(game.game_id),
            int(game.home_team_id),
            int(game.away_team_id),
        )
        if built is None:
            invalid += 1
            continue
        built.insert(0, "game_id", game.game_id)
        frames.append(built)
    if not frames:
        raise FileNotFoundError("no lineup snapshots could be built")
    lineups = pd.concat(frames, ignore_index=True)
    print(
        f"lineups rows={len(lineups)} games={lineups['game_id'].nunique()} invalid={invalid} missing={missing}",
        flush=True,
    )
    return lineups


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download game rotations and build on-court features.")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_download:
        ingest_rotations()
    lineups = build_lineups()
    destination = PROCESSED_DIR / "lineups.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lineups.to_parquet(destination, index=False)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise

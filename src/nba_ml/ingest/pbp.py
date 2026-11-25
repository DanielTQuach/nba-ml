"""Play-by-play download and a normalized event table."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ml.config import MAX_RETRIES, PBP_DIR, REQUEST_SLEEP_SECONDS, REQUEST_TIMEOUT, SEASONS
from nba_ml.ingest.game_logs import load_games, normalize_game_id

ISO_CLOCK = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
MMSS_CLOCK = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")
SUB_DESCRIPTION = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$", re.IGNORECASE)
DESC_PLAYER = re.compile(
    r"^(?:MISS\s+)?((?:[A-Z]\.\s+)?[A-Z][A-Za-z'’.\-]*(?:\s+(?:Jr\.|Sr\.|II|III|IV))?)"
)

# Foul subtypes that do not add a team foul toward the bonus.
_NON_TEAM_FOUL = ("offensive", "technical", "double")


def parse_clock(value: object) -> float:
    """Seconds remaining in the period. Accepts '5:32' and 'PT05M32.00S'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("missing clock")
    text = str(value).strip()
    match = MMSS_CLOCK.match(text)
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    match = ISO_CLOCK.match(text)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return minutes * 60 + seconds
    raise ValueError(f"unrecognized clock {value!r}")


def counts_as_team_foul(action_type: object, sub_type: object) -> bool:
    if str(action_type).strip().lower() != "foul":
        return False
    text = str(sub_type or "").strip().lower()
    if any(token in text for token in _NON_TEAM_FOUL):
        return False
    return True


def _blank_to_nan(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() == "nan" or text.lower() == "none":
        return np.nan
    return float(text)


def _optional_int(value: object) -> float:
    number = _blank_to_nan(value)
    if np.isnan(number):
        return np.nan
    return float(int(number))


def parse_substitution(description: object) -> tuple[str | None, str | None]:
    if description is None or (isinstance(description, float) and np.isnan(description)):
        return None, None
    match = SUB_DESCRIPTION.search(str(description).strip())
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _orient_score_columns(frame: pd.DataFrame, home_pts: float, away_pts: float) -> pd.DataFrame:
    """Make score_home / score_away match the game log, whichever order the feed used."""
    scored = frame.dropna(subset=["score_first", "score_second"])
    if scored.empty:
        raise ValueError("play-by-play has no score")
    last = scored.iloc[-1]
    first, second = float(last["score_first"]), float(last["score_second"])
    if first == float(home_pts) and second == float(away_pts):
        home_col, away_col = "score_first", "score_second"
    elif first == float(away_pts) and second == float(home_pts):
        home_col, away_col = "score_second", "score_first"
    else:
        raise ValueError(
            f"final play-by-play score {first}-{second} does not match game log {home_pts}-{away_pts}"
        )
    frame = frame.copy()
    frame["score_home"] = frame[home_col]
    frame["score_away"] = frame[away_col]
    return frame


def _from_v3(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "action_number": pd.to_numeric(raw["actionNumber"], errors="coerce"),
            "period": pd.to_numeric(raw["period"], errors="coerce").astype(int),
            "seconds_left": raw["clock"].map(parse_clock),
            "action_type": raw["actionType"].astype(str).str.lower(),
            "sub_type": raw["subType"].astype(str).str.lower().replace({"nan": "", "none": ""}),
            "team_id": raw["teamId"].map(_optional_int) if "teamId" in raw.columns else np.nan,
            "person_id": raw["personId"].map(_optional_int) if "personId" in raw.columns else np.nan,
            "player_name": raw["playerName"] if "playerName" in raw.columns else "",
            "description": raw["description"] if "description" in raw.columns else "",
            "score_first": raw["scoreHome"].map(_blank_to_nan),
            "score_second": raw["scoreAway"].map(_blank_to_nan),
        }
    )
    incoming_names = []
    outgoing_names = []
    person_out = []
    for description, action_type, person_id in zip(
        frame["description"], frame["action_type"], frame["person_id"], strict=True
    ):
        in_name, out_name = parse_substitution(description)
        if action_type == "substitution":
            # On V3 the row's player is the one leaving. The incoming player is only named.
            incoming_names.append(in_name)
            outgoing_names.append(out_name)
            person_out.append(person_id)
        else:
            incoming_names.append(None)
            outgoing_names.append(None)
            person_out.append(np.nan)
    frame["person_in"] = np.nan
    frame["person_out"] = person_out
    frame["player_in_name"] = incoming_names
    frame["player_out_name"] = outgoing_names
    return frame


def _v2_action(event_type: int) -> str:
    return {
        1: "made_shot",
        2: "missed_shot",
        3: "freethrow",
        4: "rebound",
        5: "turnover",
        6: "foul",
        7: "violation",
        8: "substitution",
        9: "timeout",
        10: "jumpball",
        12: "period",
        13: "period",
    }.get(int(event_type), f"event_{event_type}")


def _v2_foul_subtype(action: int) -> str:
    # Common stats.nba.com EVENTMSGACTIONTYPE values for fouls.
    return {
        1: "personal",
        2: "shooting",
        3: "looseball",
        4: "offensive",
        11: "technical",
        12: "non-unsportsmanlike technical",
        13: "hanging technical",
        14: "flagrant-type-1",
        15: "flagrant-type-2",
        16: "double technical",
        26: "offensive charge",
        27: "personal block",
        28: "shooting block",
        29: "away from play",
    }.get(int(action), str(action))


def _from_v2(raw: pd.DataFrame) -> pd.DataFrame:
    descriptions = []
    for home_desc, visitor_desc, neutral_desc in zip(
        raw.get("HOMEDESCRIPTION", pd.Series([""] * len(raw))),
        raw.get("VISITORDESCRIPTION", pd.Series([""] * len(raw))),
        raw.get("NEUTRALDESCRIPTION", pd.Series([""] * len(raw))),
        strict=False,
    ):
        text = next((str(part) for part in (home_desc, visitor_desc, neutral_desc) if pd.notna(part) and str(part).strip()), "")
        descriptions.append(text)
    event_type = pd.to_numeric(raw["EVENTMSGTYPE"], errors="coerce").fillna(-1).astype(int)
    action = pd.to_numeric(raw.get("EVENTMSGACTIONTYPE", 0), errors="coerce").fillna(0).astype(int)
    score_parts = raw["SCORE"].astype(str).str.split("-", n=1, expand=True) if "SCORE" in raw.columns else None
    if score_parts is None or score_parts.shape[1] < 2:
        score_first = pd.Series(np.nan, index=raw.index)
        score_second = pd.Series(np.nan, index=raw.index)
    else:
        score_first = score_parts[0].map(_blank_to_nan)
        score_second = score_parts[1].map(_blank_to_nan)
    sub_type = [
        _v2_foul_subtype(action_code) if event == 6 else ""
        for event, action_code in zip(event_type, action, strict=True)
    ]
    frame = pd.DataFrame(
        {
            "action_number": pd.to_numeric(raw["EVENTNUM"], errors="coerce"),
            "period": pd.to_numeric(raw["PERIOD"], errors="coerce").astype(int),
            "seconds_left": raw["PCTIMESTRING"].map(parse_clock),
            "action_type": [_v2_action(event) for event in event_type],
            "sub_type": sub_type,
            "team_id": raw["PLAYER1_TEAM_ID"].map(_optional_int) if "PLAYER1_TEAM_ID" in raw.columns else np.nan,
            "person_id": raw["PLAYER1_ID"].map(_optional_int) if "PLAYER1_ID" in raw.columns else np.nan,
            "player_name": raw["PLAYER1_NAME"] if "PLAYER1_NAME" in raw.columns else "",
            "description": descriptions,
            "score_first": score_first,
            "score_second": score_second,
            "person_in": np.nan,
            "person_out": np.nan,
            "player_out_name": None,
        }
    )
    # V2 substitutions: player1 leaves, player2 enters.
    is_sub = frame["action_type"] == "substitution"
    if "PLAYER2_ID" in raw.columns:
        frame.loc[is_sub, "person_in"] = raw.loc[is_sub, "PLAYER2_ID"].map(_optional_int).to_numpy()
        frame.loc[is_sub, "person_id"] = frame.loc[is_sub, "person_in"]
    if "PLAYER1_ID" in raw.columns:
        frame.loc[is_sub, "person_out"] = raw.loc[is_sub, "PLAYER1_ID"].map(_optional_int).to_numpy()
    if "PLAYER1_NAME" in raw.columns:
        frame.loc[is_sub, "player_out_name"] = raw.loc[is_sub, "PLAYER1_NAME"].to_numpy()
    return frame


def _lookup_player_id(
    name: object,
    team_id: object,
    by_team: dict[tuple[int, str], set[int]],
    overall: dict[str, set[int]],
) -> float:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return np.nan
    key = str(name).strip().upper()
    if not key or key == "NAN":
        return np.nan
    keys = [key, key.split()[-1]]
    if not pd.isna(team_id):
        for candidate in keys:
            matches = by_team.get((int(team_id), candidate), set())
            if len(matches) == 1:
                return float(next(iter(matches)))
    for candidate in keys:
        matches = overall.get(candidate, set())
        if len(matches) == 1:
            return float(next(iter(matches)))
    return np.nan


def _add_player_alias(
    by_team: dict[tuple[int, str], set[int]],
    overall: dict[str, set[int]],
    team_id: object,
    alias: str,
    player_id: int,
) -> None:
    key = alias.strip().upper()
    if not key or key == "NAN":
        return
    overall.setdefault(key, set()).add(player_id)
    overall.setdefault(key.split()[-1], set()).add(player_id)
    if not pd.isna(team_id):
        team_key = int(team_id)
        by_team.setdefault((team_key, key), set()).add(player_id)
        by_team.setdefault((team_key, key.split()[-1]), set()).add(player_id)


def _resolve_substitution_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill incoming player ids. Descriptions disambiguate shared last names like J. Green and D. Green."""
    by_team: dict[tuple[int, str], set[int]] = {}
    overall: dict[str, set[int]] = {}
    for person_id, player_name, team_id, action_type, description in zip(
        frame["person_id"],
        frame["player_name"],
        frame["team_id"],
        frame["action_type"],
        frame["description"],
        strict=True,
    ):
        if pd.isna(person_id):
            continue
        player_id = int(person_id)
        if player_name is not None and str(player_name).strip():
            _add_player_alias(by_team, overall, team_id, str(player_name), player_id)
        if action_type == "substitution":
            continue
        match = DESC_PLAYER.match(str(description or "").strip())
        if not match:
            continue
        alias = match.group(1).strip()
        player_text = str(player_name or "").strip()
        player_last = player_text.upper().split()[-1] if player_text else ""
        if player_last and player_last not in alias.upper():
            continue
        _add_player_alias(by_team, overall, team_id, alias, player_id)

    frame = frame.copy()
    if "player_in_name" not in frame.columns:
        frame["player_in_name"] = None
    resolved_in = []
    for current, incoming_name, team_id in zip(
        frame["person_in"], frame["player_in_name"], frame["team_id"], strict=True
    ):
        if pd.notna(current):
            resolved_in.append(float(current))
        else:
            resolved_in.append(_lookup_player_id(incoming_name, team_id, by_team, overall))
    frame["person_in"] = resolved_in
    return frame


def normalize_pbp(raw: pd.DataFrame, home_pts: float, away_pts: float) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("empty play-by-play")
    if "actionType" in raw.columns:
        frame = _from_v3(raw)
    elif "EVENTMSGTYPE" in raw.columns:
        frame = _from_v2(raw)
    else:
        raise ValueError(f"unrecognized play-by-play columns: {list(raw.columns)[:12]}")

    frame = frame.sort_values(["period", "seconds_left", "action_number"], ascending=[True, False, True])
    frame = _resolve_substitution_ids(frame)
    frame = _orient_score_columns(frame, home_pts, away_pts)
    frame["score_home"] = frame["score_home"].ffill().fillna(0.0)
    frame["score_away"] = frame["score_away"].ffill().fillna(0.0)
    frame["is_team_foul"] = [
        counts_as_team_foul(action, sub)
        for action, sub in zip(frame["action_type"], frame["sub_type"], strict=True)
    ]
    keep = [
        "action_number",
        "period",
        "seconds_left",
        "action_type",
        "sub_type",
        "team_id",
        "person_id",
        "player_name",
        "person_in",
        "person_out",
        "player_out_name",
        "player_in_name",
        "description",
        "score_home",
        "score_away",
        "is_team_foul",
    ]
    return frame[keep].reset_index(drop=True)


def _fetch_raw_pbp(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import PlayByPlayV2, PlayByPlayV3

    game_id = normalize_game_id(game_id)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            v3_endpoint = PlayByPlayV3(
                game_id=game_id,
                start_period=1,
                end_period=14,
                timeout=REQUEST_TIMEOUT,
            )
            v3 = v3_endpoint.play_by_play.get_data_frame()
            if v3 is not None and not v3.empty:
                return v3
            v2_endpoint = PlayByPlayV2(
                game_id=game_id,
                start_period=1,
                end_period=14,
                timeout=REQUEST_TIMEOUT,
            )
            v2 = v2_endpoint.play_by_play.get_data_frame()
            if v2 is not None and not v2.empty:
                return v2
            raise RuntimeError("both play-by-play endpoints returned no rows")
        except Exception as exc:  # noqa: BLE001 - network and API errors retry together
            last_error = exc
            time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise RuntimeError(f"play-by-play for {game_id} failed after {MAX_RETRIES} tries") from last_error


def pbp_path(game_id: str, directory: Path | None = None) -> Path:
    directory = directory or PBP_DIR
    return directory / f"{normalize_game_id(game_id)}.parquet"


def load_pbp(game_id: str, directory: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(pbp_path(game_id, directory))


def ingest_pbp(seasons: list[str] | None = None, max_games: int | None = None) -> dict[str, int]:
    seasons = seasons or list(SEASONS)
    games = load_games()
    selected = games[games["season"].isin(seasons)].copy()
    if max_games is not None:
        selected = selected.head(max_games)
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    cached = 0
    downloaded = 0
    failed = 0
    total = len(selected)
    for index, game in enumerate(selected.itertuples(index=False), start=1):
        destination = pbp_path(game.game_id)
        if destination.exists() and destination.stat().st_size > 0:
            cached += 1
            continue
        try:
            raw = _fetch_raw_pbp(game.game_id)
            normalized = normalize_pbp(raw, home_pts=game.home_pts, away_pts=game.away_pts)
            normalized.to_parquet(destination, index=False)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - one bad game should not stop the cache
            failed += 1
            print(f"failed {game.game_id}: {exc}", flush=True)
        if index % 25 == 0 or index == total:
            print(
                f"pbp {index}/{total} downloaded={downloaded} cached={cached} failed={failed}",
                flush=True,
            )
    print(f"pbp done downloaded={downloaded} cached={cached} failed={failed}", flush=True)
    return {"downloaded": downloaded, "cached": cached, "failed": failed, "total": total}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and cache normalized play-by-play.")
    parser.add_argument("--season", action="append", dest="seasons")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args(argv)
    ingest_pbp(args.seasons, args.max_games)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise

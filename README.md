# nba-ml

Probability that an NBA game ends at a combined final score, given the current clock, score, teams, and (in the second model) the players on the court.

The model learns how many points are still to be scored. The probability the final total equals X is the probability that the remaining points equal X minus the current total.

Training uses completed regular seasons 2022-23 through 2025-26. The 2021-22 season is only a prior for pregame team ratings. The in-progress 2026-27 season is not used.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Downloaded stats, processed tables, checkpoints, and plots are written under `data/` and are not committed.

## Pipeline

```powershell
python -m nba_ml.ingest.game_logs
python -m nba_ml.ingest.pbp
python -m nba_ml.features.snapshots
python -m nba_ml.models.train --phase 1
python -m nba_ml.eval.report
python -m nba_ml.features.lineups
python -m nba_ml.models.train --phase 2
python -m nba_ml.eval.report --phase 2
python -m nba_ml.predict --state examples/state.json --x 220
```

`ingest` calls are resumable. A game whose play-by-play file is already cached is skipped.

`predict` loads the phase 1 checkpoint. On the held-out half of 2025-26, adding on-court players did not beat that model. Pass `--checkpoint data/models/phase2.pt` to score the lineup model instead.

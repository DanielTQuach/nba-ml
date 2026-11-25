"""Negative-binomial network for points still to be scored."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import NegativeBinomial

from nba_ml.config import REMAINING_PMF_MAX


def negative_binomial(mu: torch.Tensor, alpha: torch.Tensor) -> NegativeBinomial:
    """NB2 parameterization: mean mu, variance mu + alpha * mu^2.

    PyTorch's NegativeBinomial mean is total_count * probs / (1 - probs).
    """
    total_count = (1.0 / alpha).clamp(1e-3, 1e4)
    probs = (mu / (mu + total_count)).clamp(1e-4, 1.0 - 1e-4)
    return NegativeBinomial(total_count, probs=probs)


class RemainingPointsNet(nn.Module):
    def __init__(
        self,
        n_teams: int,
        numeric_dim: int,
        n_players: int = 1,
        use_lineups: bool = False,
        team_dim: int = 16,
        player_dim: int = 16,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.use_lineups = use_lineups
        self.team_embedding = nn.Embedding(n_teams, team_dim)
        width = numeric_dim + team_dim * 2
        if use_lineups:
            self.player_embedding = nn.Embedding(n_players, player_dim, padding_idx=0)
            width += player_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        home_team: torch.Tensor,
        away_team: torch.Tensor,
        players: torch.Tensor | None = None,
        player_minutes: torch.Tensor | None = None,
        player_fouls: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [numeric, self.team_embedding(home_team), self.team_embedding(away_team)]
        if self.use_lineups:
            if players is None or player_minutes is None or player_fouls is None:
                raise ValueError("lineup tensors are required for this checkpoint")
            embedded = self.player_embedding(players)
            token = torch.cat(
                [embedded, player_minutes.unsqueeze(-1), player_fouls.unsqueeze(-1)],
                dim=-1,
            )
            parts.append(token.sum(dim=1))
        raw = self.mlp(torch.cat(parts, dim=-1))
        mu = torch.nn.functional.softplus(raw[:, 0]) + 1e-3
        alpha = torch.nn.functional.softplus(raw[:, 1]) + 1e-3
        return mu, alpha


def negative_log_likelihood(mu: torch.Tensor, alpha: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    distribution = negative_binomial(mu, alpha)
    return -distribution.log_prob(remaining.float()).mean()


def remaining_pmf(mu: torch.Tensor, alpha: torch.Tensor, max_count: int = REMAINING_PMF_MAX) -> torch.Tensor:
    """Probability of each remaining total from 0 through max_count, per row."""
    distribution = negative_binomial(mu.unsqueeze(-1), alpha.unsqueeze(-1))
    counts = torch.arange(0, max_count + 1, device=mu.device, dtype=mu.dtype).view(1, -1)
    counts = counts.expand(mu.shape[0], -1)
    probability = distribution.log_prob(counts).exp()
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-12)


def shift_to_final_total(remaining_probability: torch.Tensor | np.ndarray, current_total: int) -> np.ndarray:
    """Place remaining-point mass on absolute final totals. Index is the final total."""
    probability = np.asarray(remaining_probability, dtype=float).reshape(-1)
    final = np.zeros(current_total + probability.shape[0], dtype=float)
    final[current_total : current_total + probability.shape[0]] = probability
    return final

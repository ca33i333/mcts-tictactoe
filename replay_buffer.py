"""
replay_buffer.py
────────────────
Stores game trajectories produced by self-play for training.

Structure mirrors the MuZero paper:
  • A game trajectory = sequence of (observation, action, reward, policy, value_target).
  • Sampling is uniform (or prioritised — stub provided).
  • The buffer keeps the most recent `capacity` games.

Key types
─────────
  Transition   — one time-step inside a game.
  GameRecord   — full game trajectory.
  ReplayBuffer — circular buffer of GameRecords with batch sampling.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
from environment import NUM_ACTIONS


# ── Single time-step ──────────────────────────────────────────────────────────
@dataclass
class Transition:
    """
    One step of a real game trajectory (produced by self-play / human play).

    observation:    (3, 3, 3) float32  — what the agent saw before acting.
    action:         int                — action taken.
    reward:         float              — immediate reward received.
    policy_targets: (9,) float32       — MCTS visit-count distribution (π_MCTS).
    value_target:   float              — bootstrapped return (used as training target for f()).
    to_play:        int                — which player acted (+1 / -1).
    legal_actions:  List[int]          — legal actions at this step.
    done:           bool               — was this the terminal step?
    """
    observation:    np.ndarray
    action:         int
    reward:         float
    policy_targets: np.ndarray
    value_target:   float
    to_play:        int
    legal_actions:  List[int]
    done:           bool


# ── Full game record ──────────────────────────────────────────────────────────
@dataclass
class GameRecord:
    """
    Complete trajectory from a single episode.
    After the game finishes, call `compute_value_targets()` to fill in
    bootstrapped returns for every Transition.
    """
    transitions: List[Transition] = field(default_factory=list)
    winner:      Optional[int]    = None   # PLAYER_X, PLAYER_O, 0 (draw), None
    game_id:     int              = 0

    # ── Add a step ────────────────────────────────────────────────────────────
    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    # ── Compute bootstrap targets after game ends ──────────────────────────────
    def compute_value_targets(self, discount: float = 1.0) -> None:
        """
        Fill in value_target for each transition using discounted returns.
        Value targets are from the perspective of the player who acted.
        """
        T = len(self.transitions)
        # Work backwards: G_t = r_t + γ·G_{t+1}
        # Rewards are already from the acting player's perspective.
        cumulative = 0.0
        for t in reversed(range(T)):
            tr = self.transitions[t]
            cumulative = tr.reward + discount * cumulative
            # Flip if the next step is the opponent's perspective
            # (handled by alternating signs in self-play)
            self.transitions[t] = Transition(
                observation=tr.observation,
                action=tr.action,
                reward=tr.reward,
                policy_targets=tr.policy_targets,
                value_target=cumulative,
                to_play=tr.to_play,
                legal_actions=tr.legal_actions,
                done=tr.done,
            )
            cumulative = -cumulative   # flip for previous player (zero-sum)

    # ── Metadata ──────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.transitions)

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    def result_str(self) -> str:
        if self.winner is None:  return "ongoing"
        if self.winner ==  0:    return "draw"
        if self.winner ==  1:    return "X wins"
        return "O wins"


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Circular buffer of GameRecords.

    Supports:
      • Adding completed games.
      • Sampling random mini-batches of Transitions for training.
      • Basic statistics reporting.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity  = capacity
        self._games:   deque[GameRecord] = deque(maxlen=capacity)
        self._game_id: int = 0
        self._total_transitions: int = 0

    # ── Write ─────────────────────────────────────────────────────────────────
    def add_game(self, record: GameRecord) -> None:
        record.game_id = self._game_id
        self._game_id += 1
        self._total_transitions += len(record)
        self._games.append(record)

    # ── Read ──────────────────────────────────────────────────────────────────
    def sample_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample `batch_size` individual Transitions uniformly at random
        across all stored games.
        """
        all_transitions = [t for game in self._games for t in game.transitions]
        if len(all_transitions) == 0:
            return []
        idx = np.random.choice(len(all_transitions), size=min(batch_size, len(all_transitions)), replace=False)
        return [all_transitions[i] for i in idx]

    def sample_game(self) -> Optional[GameRecord]:
        """Sample a single game uniformly."""
        if not self._games:
            return None
        return self._games[np.random.randint(len(self._games))]

    def sample_unroll_batch(
        self,
        batch_size:  int,
        unroll_steps: int = 5,
    ) -> List[List[Transition]]:
        """
        Sample `batch_size` sub-trajectories each of length `unroll_steps`
        (padded at the end with the terminal transition if needed).
        This is the standard MuZero training input.
        """
        batch = []
        games = [g for g in self._games if len(g) >= 1]
        if not games:
            return batch

        for _ in range(batch_size):
            game = games[np.random.randint(len(games))]
            start = np.random.randint(len(game))
            end   = min(start + unroll_steps, len(game))
            sub   = game.transitions[start:end]
            # Pad with last transition if shorter than unroll_steps
            while len(sub) < unroll_steps:
                sub = sub + [sub[-1]]
            batch.append(sub)

        return batch

    # ── Iterators ─────────────────────────────────────────────────────────────
    def iter_games(self) -> Iterator[GameRecord]:
        return iter(self._games)

    def iter_transitions(self) -> Iterator[Transition]:
        for game in self._games:
            yield from game.transitions

    # ── Stats ─────────────────────────────────────────────────────────────────
    @property
    def num_games(self) -> int:
        return len(self._games)

    @property
    def num_transitions(self) -> int:
        return sum(len(g) for g in self._games)

    def win_rate(self) -> dict:
        wins = {1: 0, -1: 0, 0: 0, None: 0}
        for g in self._games:
            wins[g.winner] = wins.get(g.winner, 0) + 1
        total = len(self._games) or 1
        return {
            "X_wins":  wins[1]    / total,
            "O_wins":  wins[-1]   / total,
            "draws":   wins[0]    / total,
            "ongoing": wins[None] / total,
            "total_games": len(self._games),
        }

    def avg_game_length(self) -> float:
        if not self._games:
            return 0.0
        return sum(len(g) for g in self._games) / len(self._games)

    def __len__(self) -> int:
        return len(self._games)

    def __repr__(self) -> str:
        wr = self.win_rate()
        return (
            f"ReplayBuffer(games={self.num_games}/{self.capacity}, "
            f"transitions={self.num_transitions}, "
            f"X={wr['X_wins']:.1%} D={wr['draws']:.1%} O={wr['O_wins']:.1%})"
        )

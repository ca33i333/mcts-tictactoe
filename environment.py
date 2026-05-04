"""
environment.py
──────────────
Pure TicTacToe game rules. Zero ML here — just the real MDP.
The world model / MCTS never import this during planning;
it is only used to:
  • generate real trajectories for the replay buffer
  • verify correctness in tests
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
BOARD_SIZE = 3
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE   # 9

PLAYER_X =  1
PLAYER_O = -1
EMPTY    =  0

WIN_LINES: List[Tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diags
]


# ── Core result type ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StepResult:
    next_state: "TicTacToeState"
    reward:     float          # from the acting player's perspective
    done:       bool
    winner:     Optional[int]  # PLAYER_X, PLAYER_O, or 0 (draw), None if ongoing


# ── Immutable board state ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class TicTacToeState:
    """
    Immutable snapshot of the game.
    board: flat np.ndarray of shape (9,) with values in {-1, 0, 1}.
    to_play: which player acts next (PLAYER_X or PLAYER_O).
    """
    board:   Tuple[int, ...]        # length-9 tuple for hashability
    to_play: int = PLAYER_X

    # ── Constructors ──────────────────────────────────────────────────────────
    @classmethod
    def initial(cls) -> "TicTacToeState":
        return cls(board=tuple([EMPTY] * NUM_ACTIONS), to_play=PLAYER_X)

    # ── Observation (what the world model / agent sees) ───────────────────────
    def to_observation(self) -> np.ndarray:
        """
        Returns a (3, 3, 3) float32 tensor:
          channel 0 — cells occupied by current player
          channel 1 — cells occupied by opponent
          channel 2 — constant plane encoding whose turn it is
        This is the standard MuZero observation format.
        """
        b = np.array(self.board, dtype=np.float32).reshape(3, 3)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[0] = (b == self.to_play).astype(np.float32)
        obs[1] = (b == -self.to_play).astype(np.float32)
        obs[2] = np.full((3, 3), 0.5 if self.to_play == PLAYER_X else -0.5)
        return obs

    # ── Legal actions ─────────────────────────────────────────────────────────
    def legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def is_legal(self, action: int) -> bool:
        return 0 <= action < NUM_ACTIONS and self.board[action] == EMPTY

    # ── Win / terminal checks ─────────────────────────────────────────────────
    def winner(self) -> Optional[int]:
        b = self.board
        for a, c, e in WIN_LINES:
            if b[a] != EMPTY and b[a] == b[c] == b[e]:
                return b[a]
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or all(v != EMPTY for v in self.board)

    def is_draw(self) -> bool:
        return self.winner() is None and all(v != EMPTY for v in self.board)

    # ── Transition ────────────────────────────────────────────────────────────
    def step(self, action: int) -> StepResult:
        if not self.is_legal(action):
            raise ValueError(f"Illegal action {action} on board {self.board}")

        new_board = list(self.board)
        new_board[action] = self.to_play
        new_board = tuple(new_board)
        next_state = TicTacToeState(board=new_board, to_play=-self.to_play)

        w = next_state.winner()
        if w is not None:
            # reward is from the perspective of the player who just moved
            reward = 1.0 if w == self.to_play else -1.0
            return StepResult(next_state, reward, done=True, winner=w)
        if next_state.is_draw():
            return StepResult(next_state, 0.0, done=True, winner=0)
        return StepResult(next_state, 0.0, done=False, winner=None)

    # ── Display ───────────────────────────────────────────────────────────────
    def render(self) -> str:
        sym = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(sym[self.board[r*3+c]] for c in range(3)))
        header = f"To play: {'X' if self.to_play == PLAYER_X else 'O'}"
        return header + "\n" + "\n".join(rows)

    def __str__(self) -> str:
        return self.render()

    def __hash__(self):
        return hash((self.board, self.to_play))


# ── Thin environment wrapper (stateful, for game loops) ───────────────────────
class TicTacToeEnv:
    """
    Stateful wrapper around TicTacToeState for convenience in game loops.
    Mimics a gym-style interface.
    """

    def __init__(self):
        self.state: TicTacToeState = TicTacToeState.initial()
        self._history: List[TicTacToeState] = [self.state]

    def reset(self) -> np.ndarray:
        self.state = TicTacToeState.initial()
        self._history = [self.state]
        return self.state.to_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        result = self.state.step(action)
        self.state = result.next_state
        self._history.append(self.state)
        info = {"winner": result.winner, "legal_actions": self.state.legal_actions()}
        return result.next_state.to_observation(), result.reward, result.done, info

    @property
    def legal_actions(self) -> List[int]:
        return self.state.legal_actions()

    @property
    def to_play(self) -> int:
        return self.state.to_play

    def render(self) -> None:
        print(self.state.render())

    def history(self) -> List[TicTacToeState]:
        return list(self._history)

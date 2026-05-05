"""
MCTS Experience Log
====================
Captures tree nodes, rollouts, and backprop updates as structured
experience entries, then serialises them to JSON for storage / replay.

Usage
-----
    log = ExperienceLog()

    # During your MCTS loop:
    log.record_selection(path=[0, 2, 1], uct_scores=[1.41, 0.98, 1.73])
    log.record_expansion(parent_id=1, new_node_id=5, action="move_right")
    log.record_simulation(node_id=5, reward=1.0, depth=12, moves=["a","b"])
    log.record_backprop(node_id=5, path=[0, 2, 1, 5], rewards=[1.0]*4)

    # Save / load:
    log.save("experience.json")
    log2 = ExperienceLog.load("experience.json")
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────
# Entry dataclasses  (one per MCTS phase)
# ─────────────────────────────────────────────

@dataclass
class SelectionEntry:
    entry_id: str
    phase: str = "selection"
    timestamp: float = field(default_factory=time.time)
    # ids of nodes visited root → leaf
    path: list[int] = field(default_factory=list)
    # UCT score at each step
    uct_scores: list[float] = field(default_factory=list)
    selected_node_id: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionEntry:
    entry_id: str
    phase: str = "expansion"
    timestamp: float = field(default_factory=time.time)
    parent_node_id: int = -1
    new_node_id: int = -1
    action: Any = None          # whatever your action space uses
    prior_prob: float = 0.0     # optional policy-network prior
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationEntry:
    entry_id: str
    phase: str = "simulation"
    timestamp: float = field(default_factory=time.time)
    start_node_id: int = -1
    reward: float = 0.0
    rollout_depth: int = 0
    move_sequence: list[Any] = field(default_factory=list)
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackpropEntry:
    entry_id: str
    phase: str = "backprop"
    timestamp: float = field(default_factory=time.time)
    # leaf → root
    path: list[int] = field(default_factory=list)
    reward: float = 0.0
    # Q and N values *after* the update for each node in path
    updated_q: list[float] = field(default_factory=list)
    updated_n: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Experience log
# ─────────────────────────────────────────────

class ExperienceLog:
    """
    Accumulates MCTS experience entries and converts them to / from JSON.

    Each entry is stored in `self.entries` as a plain dict so the log
    stays serialisable at all times.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id: str = run_id or str(uuid.uuid4())
        self.created_at: float = time.time()
        self.entries: list[dict] = []

    # ── helpers ──────────────────────────────

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    def _add(self, entry_dc) -> str:
        d = asdict(entry_dc)
        self.entries.append(d)
        return d["entry_id"]

    # ── recording API ─────────────────────────

    def record_selection(
        self,
        path: list[int],
        uct_scores: list[float] | None = None,
        selected_node_id: int = -1,
        metadata: dict | None = None,
    ) -> str:
        e = SelectionEntry(
            entry_id=self._new_id(),
            path=path,
            uct_scores=uct_scores or [],
            selected_node_id=selected_node_id if selected_node_id != -1
                             else (path[-1] if path else -1),
            metadata=metadata or {},
        )
        return self._add(e)

    def record_expansion(
        self,
        parent_id: int,
        new_node_id: int,
        action: Any = None,
        prior_prob: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        e = ExpansionEntry(
            entry_id=self._new_id(),
            parent_node_id=parent_id,
            new_node_id=new_node_id,
            action=action,
            prior_prob=prior_prob,
            metadata=metadata or {},
        )
        return self._add(e)

    def record_simulation(
        self,
        node_id: int,
        reward: float,
        depth: int = 0,
        moves: list[Any] | None = None,
        terminal: bool = False,
        metadata: dict | None = None,
    ) -> str:
        e = SimulationEntry(
            entry_id=self._new_id(),
            start_node_id=node_id,
            reward=reward,
            rollout_depth=depth,
            move_sequence=moves or [],
            terminal=terminal,
            metadata=metadata or {},
        )
        return self._add(e)

    def record_backprop(
        self,
        path: list[int],
        reward: float,
        updated_q: list[float] | None = None,
        updated_n: list[int] | None = None,
        metadata: dict | None = None,
    ) -> str:
        e = BackpropEntry(
            entry_id=self._new_id(),
            path=path,
            reward=reward,
            updated_q=updated_q or [],
            updated_n=updated_n or [],
            metadata=metadata or {},
        )
        return self._add(e)

    # ── serialisation ─────────────────────────

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "entry_count": len(self.entries),
            "entries": self.entries,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")
        print(f"[ExperienceLog] saved {len(self.entries)} entries → {path}")

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceLog":
        log = cls(run_id=data.get("run_id"))
        log.created_at = data.get("created_at", log.created_at)
        log.entries = data.get("entries", [])
        return log

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLog":
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))

    # ── querying ──────────────────────────────

    def by_phase(self, phase: str) -> list[dict]:
        """Return all entries for a given phase."""
        return [e for e in self.entries if e.get("phase") == phase]

    def summary(self) -> dict:
        phases = ["selection", "expansion", "simulation", "backprop"]
        return {
            "run_id": self.run_id,
            "total_entries": len(self.entries),
            **{p: len(self.by_phase(p)) for p in phases},
        }

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"ExperienceLog(run_id={self.run_id!r}, entries={len(self)})"


# ─────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import math, random

    log = ExperienceLog(run_id="demo-run-001")

    # Simulate 5 MCTS iterations
    for i in range(5):
        path = list(range(random.randint(1, 4)))
        uct = [round(random.uniform(0.5, 2.0), 4) for _ in path]

        log.record_selection(path=path, uct_scores=uct)

        parent = path[-1]
        new_node = 100 + i
        log.record_expansion(
            parent_id=parent,
            new_node_id=new_node,
            action=f"action_{i}",
            prior_prob=round(random.random(), 3),
        )

        reward = round(random.choice([-1.0, 0.0, 1.0]), 1)
        moves = [f"m{j}" for j in range(random.randint(2, 6))]
        log.record_simulation(
            node_id=new_node,
            reward=reward,
            depth=len(moves),
            moves=moves,
            terminal=reward != 0,
        )

        full_path = path + [new_node]
        q_vals = [round(random.uniform(0, 1), 3) for _ in full_path]
        n_vals = [random.randint(1, 20) for _ in full_path]
        log.record_backprop(
            path=full_path,
            reward=reward,
            updated_q=q_vals,
            updated_n=n_vals,
        )

    # Print summary
    print(log.summary())

    # Save to JSON
    out = Path("/mnt/user-data/outputs/experience.json")
    log.save(out)

    # Round-trip check
    log2 = ExperienceLog.load(out)
    assert len(log2) == len(log), "Round-trip mismatch!"
    print(f"[OK] Round-trip verified: {len(log2)} entries")

    # Show first entry
    print("\nFirst entry:")
    print(json.dumps(log2.entries[0], indent=2))

"""
mcts.py
───────
Blind Monte-Carlo Tree Search — MuZero style.

Key property: after the root is encoded with h(), the search
NEVER touches the real environment again.  Every expansion is
done through g() (dynamics) and f() (prediction) only.

Algorithm (one simulation):
  1. SELECT      — walk the tree with PUCT until a leaf is reached.
  2. EXPAND      — call g() to get the next latent state, f() to get
                   the prior policy and value estimate for the new node.
  3. BACKUP      — propagate the value back up, flipping sign at each
                   level (zero-sum).

The root prior is computed by f() on the initial latent state produced
by h(); a small Dirichlet noise is added for exploration.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from world_model import WorldModel, LatentState
from environment  import NUM_ACTIONS


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class MCTSConfig:
    num_simulations:     int   = 100
    puct_c1:             float = 1.25   # base exploration constant
    puct_c2:             float = 19652  # visit scaling (AlphaZero default)
    dirichlet_alpha:     float = 0.3    # noise concentration (0.3 for chess, higher for simple games)
    dirichlet_fraction:  float = 0.25   # fraction of root prior replaced by noise
    discount:            float = 1.0    # γ for value backup
    value_prefix_weight: float = 1.0    # weight on intermediate rewards during backup


# ── MCTS Node ─────────────────────────────────────────────────────────────────
class MCTSNode:
    """
    One node in the search tree.
    Children are indexed by action (int 0..8).
    """

    __slots__ = (
        "latent", "parent", "action_from_parent",
        "prior", "visit_count", "value_sum",
        "children", "reward", "is_expanded",
    )

    def __init__(
        self,
        latent:             Optional[LatentState],
        parent:             Optional["MCTSNode"],
        action_from_parent: Optional[int],
        prior:              float,
        reward:             float = 0.0,
    ):
        self.latent             = latent
        self.parent             = parent
        self.action_from_parent = action_from_parent
        self.prior              = prior
        self.reward             = reward
        self.visit_count:  int  = 0
        self.value_sum:  float  = 0.0
        self.children:   Dict[int, "MCTSNode"] = {}
        self.is_expanded: bool  = False

    # ── Statistics ────────────────────────────────────────────────────────────
    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return not self.is_expanded

    # ── PUCT score ────────────────────────────────────────────────────────────
    def puct_score(self, parent_visit_count: int, c1: float, c2: float) -> float:
        """
        PUCT = Q(s,a) + P(s,a) · √N(s) / (1 + N(s,a)) · (c1 + log((N(s)+c2+1)/c2))
        AlphaZero/MuZero formulation.
        """
        exploration = (
            self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
            * (c1 + math.log((parent_visit_count + c2 + 1) / c2))
        )
        return self.q_value + exploration

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action_from_parent}, "
            f"N={self.visit_count}, Q={self.q_value:.3f}, "
            f"prior={self.prior:.3f}, children={len(self.children)})"
        )


# ── Search result ─────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    """Everything the caller needs after a search."""
    action:          int                      # chosen action
    policy_probs:    np.ndarray               # (NUM_ACTIONS,) visit-count distribution
    root_value:      float                    # estimated value at root
    visit_counts:    np.ndarray               # (NUM_ACTIONS,) raw visit counts
    max_depth:       int
    total_sims:      int
    child_q_values:  np.ndarray               # (NUM_ACTIONS,) Q for each child


# ── MCTS ──────────────────────────────────────────────────────────────────────
class MCTS:
    """
    Blind MCTS planner.

    Usage:
        mcts   = MCTS(world_model, config)
        result = mcts.search(observation, legal_actions)
        action = result.action
    """

    def __init__(self, world_model: WorldModel, config: MCTSConfig = MCTSConfig()):
        self.model  = world_model
        self.cfg    = config

    # ── Public entry point ────────────────────────────────────────────────────
    def search(
        self,
        observation:   np.ndarray,
        legal_actions: List[int],
        temperature:   float = 1.0,
    ) -> SearchResult:
        """
        Run full MCTS from a real observation.

        Steps:
          1. Encode observation with h() → root latent z.
          2. Get root policy/value from f(z).
          3. Add Dirichlet noise to root prior (exploration).
          4. Run `num_simulations` simulations.
          5. Extract action from visit-count distribution.
        """
        # ── Step 1 & 2: encode + predict at root ──────────────────────────────
        root_latent = self.model.represent(observation)
        root_pred   = self.model.predict(root_latent)

        # ── Step 3: build root node, mask illegal actions ─────────────────────
        root = MCTSNode(latent=root_latent, parent=None, action_from_parent=None, prior=1.0)
        self._expand_node(root, root_pred.policy_probs, legal_actions)
        self._add_dirichlet_noise(root, legal_actions)

        # ── Step 4: simulations ───────────────────────────────────────────────
        max_depth = 0
        for _ in range(self.cfg.num_simulations):
            depth = self._simulate(root)
            max_depth = max(max_depth, depth)

        # ── Step 5: extract results ───────────────────────────────────────────
        return self._build_result(root, temperature, max_depth)

    # ── Simulation (one trajectory) ───────────────────────────────────────────
    def _simulate(self, root: MCTSNode) -> int:
        node  = root
        depth = 0
        search_path: List[MCTSNode] = [node]

        # SELECTION — follow PUCT until leaf
        while not node.is_leaf():
            action, node = self._select_child(node)
            search_path.append(node)
            depth += 1

        # EXPANSION — call g() then f()
        parent = search_path[-2] if len(search_path) > 1 else None

        if parent is not None and parent.latent is not None:
            dyn_out = self.model.step(parent.latent, node.action_from_parent)
            node.latent = dyn_out.next_latent
            node.reward = dyn_out.reward

        pred = self.model.predict(node.latent)
        # All actions treated as legal inside latent space
        # (the world model is responsible for learning illegal action avoidance)
        self._expand_node(node, pred.policy_probs, list(range(NUM_ACTIONS)))
        value = pred.value

        # BACKUP
        self._backup(search_path, value)
        return depth

    # ── PUCT child selection ──────────────────────────────────────────────────
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_score  = -math.inf
        best_action = -1
        best_child  = None

        for action, child in node.children.items():
            score = child.puct_score(node.visit_count, self.cfg.puct_c1, self.cfg.puct_c2)
            if score > best_score:
                best_score  = score
                best_action = action
                best_child  = child

        return best_action, best_child

    # ── Expand a leaf node ────────────────────────────────────────────────────
    def _expand_node(
        self,
        node:          MCTSNode,
        policy_probs:  np.ndarray,
        legal_actions: List[int],
    ) -> None:
        # Mask and renormalise over legal actions
        mask  = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in legal_actions:
            mask[a] = policy_probs[a]
        total = mask.sum()
        if total > 0:
            mask /= total
        else:
            # Uniform fallback if all priors are zero (shouldn't happen)
            for a in legal_actions:
                mask[a] = 1.0 / len(legal_actions)

        for a in legal_actions:
            node.children[a] = MCTSNode(
                latent=None,
                parent=node,
                action_from_parent=a,
                prior=float(mask[a]),
            )
        node.is_expanded = True

    # ── Dirichlet noise at root ───────────────────────────────────────────────
    def _add_dirichlet_noise(self, root: MCTSNode, legal_actions: List[int]) -> None:
        if not legal_actions:
            return
        alpha = self.cfg.dirichlet_alpha
        frac  = self.cfg.dirichlet_fraction
        noise = np.random.dirichlet([alpha] * len(legal_actions))
        for i, a in enumerate(legal_actions):
            child = root.children[a]
            child.prior = (1 - frac) * child.prior + frac * noise[i]

    # ── Value backup ──────────────────────────────────────────────────────────
    def _backup(self, search_path: List[MCTSNode], leaf_value: float) -> None:
        """
        Propagate value back up, alternating sign (zero-sum game).
        Intermediate rewards are added along the path.
        """
        value = leaf_value
        discount = self.cfg.discount
        reward_weight = self.cfg.value_prefix_weight

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum   += value
            # flip sign for opponent + accumulate discounted reward
            value = -value * discount + node.reward * reward_weight

    # ── Build SearchResult ────────────────────────────────────────────────────
    def _build_result(
        self,
        root:        MCTSNode,
        temperature: float,
        max_depth:   int,
    ) -> SearchResult:
        visits   = np.zeros(NUM_ACTIONS, dtype=np.float32)
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for action, child in root.children.items():
            visits[action]   = child.visit_count
            q_values[action] = child.q_value

        # Policy: visit-count distribution with temperature
        if temperature == 0:
            # Greedy
            best = int(np.argmax(visits))
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            probs[best] = 1.0
        else:
            v_temp = visits ** (1.0 / temperature)
            total  = v_temp.sum()
            probs  = v_temp / total if total > 0 else visits / visits.sum()

        # Sample action proportional to probs
        legal = [a for a in root.children]
        legal_probs = np.array([probs[a] for a in legal], dtype=np.float64)
        legal_probs /= legal_probs.sum()
        chosen = int(np.random.choice(legal, p=legal_probs))

        return SearchResult(
            action=chosen,
            policy_probs=probs,
            root_value=root.q_value,
            visit_counts=visits,
            max_depth=max_depth,
            total_sims=self.cfg.num_simulations,
            child_q_values=q_values,
        )

    # ── Debug: pretty-print top children ─────────────────────────────────────
    def print_tree(self, root: MCTSNode, top_k: int = 5) -> None:
        children = sorted(root.children.values(), key=lambda c: -c.visit_count)[:top_k]
        print(f"Root: N={root.visit_count}, Q={root.q_value:.3f}")
        for c in children:
            bar = "█" * int(c.visit_count / max(root.visit_count, 1) * 20)
            print(f"  action={c.action_from_parent}  N={c.visit_count:4d}  "
                  f"Q={c.q_value:+.3f}  P={c.prior:.3f}  {bar}")

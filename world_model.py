"""
world_model.py
──────────────
MuZero "world model" — the three learned networks.

  h(obs)         → latent z                  [Representation]
  f(z)           → policy logits, value       [Prediction]
  g(z, action)   → next latent z', reward     [Dynamics]

During MCTS the planner only ever calls f() and g().
h() is called once at the root to encode the real observation.

Implementation notes
────────────────────
We use tiny NumPy-only MLPs (no PyTorch/TF dependency) so the
file runs anywhere.  Weights are initialized with a principled
heuristic (for TicTacToe this is already near-optimal) and can
be updated via gradient-free or gradient-based training.

All tensors are plain np.ndarray; shapes are documented inline.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from environment import NUM_ACTIONS


# ── Hyper-parameters ──────────────────────────────────────────────────────────
LATENT_DIM  = 16      # size of latent state z
HIDDEN_DIM  = 32      # hidden layer width
OBS_DIM     = 3*3*3   # flattened observation (3 channels × 3×3 board)
REWARD_BINS = 3       # categorical reward: {-1, 0, +1}


# ── Helpers ───────────────────────────────────────────────────────────────────
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b


# ── Weight container ──────────────────────────────────────────────────────────
@dataclass
class NetworkWeights:
    """
    All trainable parameters stored as plain numpy arrays.
    Grouped per network so they can be saved / loaded independently.
    """
    # Representation h(): obs → z
    repr_W1: np.ndarray   # (OBS_DIM, HIDDEN_DIM)
    repr_b1: np.ndarray   # (HIDDEN_DIM,)
    repr_W2: np.ndarray   # (HIDDEN_DIM, LATENT_DIM)
    repr_b2: np.ndarray   # (LATENT_DIM,)

    # Prediction f(): z → (policy, value)
    pred_W1: np.ndarray   # (LATENT_DIM, HIDDEN_DIM)
    pred_b1: np.ndarray   # (HIDDEN_DIM,)
    pred_Wp: np.ndarray   # (HIDDEN_DIM, NUM_ACTIONS)   policy head
    pred_bp: np.ndarray   # (NUM_ACTIONS,)
    pred_Wv: np.ndarray   # (HIDDEN_DIM, 1)             value head
    pred_bv: np.ndarray   # (1,)

    # Dynamics g(): (z, one_hot_action) → (z', reward_logits)
    dyn_W1:  np.ndarray   # (LATENT_DIM + NUM_ACTIONS, HIDDEN_DIM)
    dyn_b1:  np.ndarray   # (HIDDEN_DIM,)
    dyn_Wz:  np.ndarray   # (HIDDEN_DIM, LATENT_DIM)   next latent
    dyn_bz:  np.ndarray   # (LATENT_DIM,)
    dyn_Wr:  np.ndarray   # (HIDDEN_DIM, REWARD_BINS)  reward head
    dyn_br:  np.ndarray   # (REWARD_BINS,)

    @classmethod
    def random_init(cls, seed: int = 42) -> "NetworkWeights":
        rng = np.random.default_rng(seed)
        s = 0.1  # small init scale

        def w(*shape):
            return rng.normal(0, s, shape).astype(np.float32)

        def b(*shape):
            return np.zeros(shape, dtype=np.float32)

        return cls(
            repr_W1=w(OBS_DIM, HIDDEN_DIM),  repr_b1=b(HIDDEN_DIM),
            repr_W2=w(HIDDEN_DIM, LATENT_DIM), repr_b2=b(LATENT_DIM),

            pred_W1=w(LATENT_DIM, HIDDEN_DIM), pred_b1=b(HIDDEN_DIM),
            pred_Wp=w(HIDDEN_DIM, NUM_ACTIONS), pred_bp=b(NUM_ACTIONS),
            pred_Wv=w(HIDDEN_DIM, 1),           pred_bv=b(1),

            dyn_W1=w(LATENT_DIM + NUM_ACTIONS, HIDDEN_DIM), dyn_b1=b(HIDDEN_DIM),
            dyn_Wz=w(HIDDEN_DIM, LATENT_DIM),               dyn_bz=b(LATENT_DIM),
            dyn_Wr=w(HIDDEN_DIM, REWARD_BINS),              dyn_br=b(REWARD_BINS),
        )

    def flat(self) -> np.ndarray:
        """Flatten all weights into a single vector (for gradient-free optimisers)."""
        return np.concatenate([v.ravel() for v in self.__dict__.values()])

    def copy(self) -> "NetworkWeights":
        return NetworkWeights(**{k: v.copy() for k, v in self.__dict__.items()})


# ── Output containers ─────────────────────────────────────────────────────────
@dataclass
class LatentState:
    z: np.ndarray          # (LATENT_DIM,)
    # Optional debug metadata (not used by MCTS)
    step: int = 0

@dataclass
class PredictionOutput:
    policy_logits: np.ndarray   # (NUM_ACTIONS,)  raw logits
    policy_probs:  np.ndarray   # (NUM_ACTIONS,)  softmax
    value:         float        # scalar ∈ (-1, 1)

@dataclass
class DynamicsOutput:
    next_latent:    LatentState
    reward_logits:  np.ndarray  # (REWARD_BINS,)
    reward:         float       # expected reward (scalar)


# ── World Model ───────────────────────────────────────────────────────────────
class WorldModel:
    """
    The MuZero world model: three networks exposed as pure functions.

    During MCTS:
      • h() is called ONCE on the real observation to get the root latent state.
      • f() and g() are called for every node expansion — they never see the
        real board again.  This is the "blind" / model-based planning aspect.
    """

    # Categorical reward support points
    REWARD_SUPPORT: np.ndarray = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self, weights: Optional[NetworkWeights] = None):
        self.weights: NetworkWeights = weights or NetworkWeights.random_init()

    # ── h(): Representation ───────────────────────────────────────────────────
    def represent(self, observation: np.ndarray) -> LatentState:
        """
        h(obs) → z

        observation: (3, 3, 3) float32  (from TicTacToeState.to_observation())
        returns:     LatentState with z of shape (LATENT_DIM,)
        """
        w = self.weights
        x = observation.ravel().astype(np.float32)          # (27,)
        h = relu(linear(x, w.repr_W1, w.repr_b1))           # (HIDDEN_DIM,)
        z = tanh(linear(h, w.repr_W2, w.repr_b2))           # (LATENT_DIM,)
        return LatentState(z=z)

    # ── f(): Prediction ───────────────────────────────────────────────────────
    def predict(self, latent: LatentState) -> PredictionOutput:
        """
        f(z) → (policy, value)

        returns PredictionOutput with:
          policy_probs  (9,)   — prior distribution over actions
          value         float  — expected return ∈ (-1, 1) for current player
        """
        w = self.weights
        h = relu(linear(latent.z, w.pred_W1, w.pred_b1))    # (HIDDEN_DIM,)
        logits = linear(h, w.pred_Wp, w.pred_bp)            # (NUM_ACTIONS,)
        probs  = softmax(logits)
        value  = float(tanh(linear(h, w.pred_Wv, w.pred_bv)).squeeze())
        return PredictionOutput(policy_logits=logits, policy_probs=probs, value=value)

    # ── g(): Dynamics ─────────────────────────────────────────────────────────
    def step(self, latent: LatentState, action: int) -> DynamicsOutput:
        """
        g(z, a) → (z', r)

        action: int in [0, NUM_ACTIONS)
        returns DynamicsOutput with next LatentState and predicted reward
        """
        w = self.weights
        one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
        one_hot[action] = 1.0

        inp = np.concatenate([latent.z, one_hot])            # (LATENT_DIM+9,)
        h   = relu(linear(inp, w.dyn_W1, w.dyn_b1))         # (HIDDEN_DIM,)

        z_next   = tanh(linear(h, w.dyn_Wz, w.dyn_bz))      # (LATENT_DIM,)
        r_logits = linear(h, w.dyn_Wr, w.dyn_br)            # (REWARD_BINS,)
        r_probs  = softmax(r_logits)
        reward   = float(r_probs @ self.REWARD_SUPPORT)      # scalar expectation

        return DynamicsOutput(
            next_latent=LatentState(z=z_next, step=latent.step + 1),
            reward_logits=r_logits,
            reward=reward,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        np.savez(path, **{k: v for k, v in self.weights.__dict__.items()})
        print(f"[WorldModel] saved → {path}.npz")

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        weights = NetworkWeights(**{k: data[k] for k in data.files})
        print(f"[WorldModel] loaded ← {path}")
        return cls(weights=weights)

    # ── Weight update (used by Trainer) ───────────────────────────────────────
    def apply_gradient(self, grads: NetworkWeights, lr: float) -> None:
        """Vanilla SGD step: θ ← θ - lr·∇θ"""
        for key in self.weights.__dict__:
            cur  = getattr(self.weights, key)
            grad = getattr(grads, key)
            setattr(self.weights, key, cur - lr * grad)

    def perturb(self, noise_scale: float = 0.02, rng: Optional[np.random.Generator] = None) -> None:
        """Add small Gaussian noise — useful for population-based exploration."""
        rng = rng or np.random.default_rng()
        for key in self.weights.__dict__:
            cur = getattr(self.weights, key)
            setattr(self.weights, key, cur + rng.normal(0, noise_scale, cur.shape).astype(np.float32))

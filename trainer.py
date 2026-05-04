"""
trainer.py
──────────
Self-play training loop for the MuZero world model.

Pipeline (one training iteration):
  1. SELF-PLAY   — play N games using MCTS + current world model,
                   store trajectories in the replay buffer.
  2. SAMPLE      — draw mini-batches of unrolled sub-trajectories.
  3. COMPUTE LOSS — three loss terms:
       • L_v  — value loss:   MSE( f(z).value,  G_t )
       • L_π  — policy loss:  CrossEntropy( f(z).policy, π_MCTS )
       • L_r  — reward loss:  CrossEntropy( g(z,a).reward_logits, r_t )
  4. UPDATE      — finite-difference gradient approximation (no autograd).

Notes
─────
• We use a finite-difference (SPSA-like) gradient estimator so the code
  stays dependency-free.  Plug in PyTorch / JAX by replacing `_compute_gradients`.
• Temperature annealing: high temp early (exploration), low temp later (exploitation).
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from environment   import TicTacToeEnv, NUM_ACTIONS, PLAYER_X, PLAYER_O
from world_model   import WorldModel, NetworkWeights, LATENT_DIM
from mcts          import MCTS, MCTSConfig, SearchResult
from replay_buffer import ReplayBuffer, GameRecord, Transition


# ── Training config ───────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Self-play
    games_per_iteration:  int   = 20
    mcts_simulations:     int   = 50
    temperature_threshold: int  = 4       # steps before switching to greedy
    temperature_init:     float = 1.0
    temperature_final:    float = 0.1

    # Training
    iterations:           int   = 30
    batch_size:           int   = 64
    unroll_steps:         int   = 5
    learning_rate:        float = 0.01
    weight_decay:         float = 1e-4

    # Loss weights
    value_loss_weight:    float = 1.0
    policy_loss_weight:   float = 1.0
    reward_loss_weight:   float = 1.0

    # SPSA gradient estimation
    spsa_epsilon:         float = 1e-2    # perturbation magnitude
    spsa_samples:         int   = 4      # number of perturbation pairs

    # Buffer
    buffer_capacity:      int   = 2000
    min_buffer_games:     int   = 5      # don't train until buffer has this many games

    # Misc
    eval_every:           int   = 5      # evaluate every N iterations
    eval_games:           int   = 20
    discount:             float = 1.0
    seed:                 int   = 0
    verbose:              bool  = True


# ── Loss functions ────────────────────────────────────────────────────────────
def cross_entropy(logits: np.ndarray, target_probs: np.ndarray, eps: float = 1e-8) -> float:
    log_probs = logits - np.log(np.exp(logits).sum() + eps)
    return -float((target_probs * log_probs).sum())

def mse(pred: float, target: float) -> float:
    return (pred - target) ** 2

def scalar_to_support(value: float, support: np.ndarray) -> np.ndarray:
    """Convert scalar reward/value to categorical distribution over support points."""
    clipped = np.clip(value, support[0], support[-1])
    lo_idx  = np.searchsorted(support, clipped, side='right') - 1
    lo_idx  = np.clip(lo_idx, 0, len(support) - 2)
    lo, hi  = support[lo_idx], support[lo_idx + 1]
    phi     = np.zeros(len(support), dtype=np.float32)
    phi[lo_idx]     = (hi - clipped) / (hi - lo + 1e-8)
    phi[lo_idx + 1] = (clipped - lo) / (hi - lo + 1e-8)
    return phi


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    """
    Orchestrates self-play → buffer → gradient update cycles.
    """

    REWARD_SUPPORT = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self, model: WorldModel, config: TrainConfig = TrainConfig()):
        self.model  = model
        self.cfg    = config
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)

        mcts_cfg = MCTSConfig(num_simulations=config.mcts_simulations)
        self.mcts   = MCTS(model, mcts_cfg)

        self.rng    = np.random.default_rng(config.seed)
        self._iteration = 0

        # Logging
        self.loss_history:   List[float] = []
        self.win_history:    List[dict]  = []

    # ── Public ────────────────────────────────────────────────────────────────
    def train(self) -> None:
        """Run all training iterations."""
        for it in range(self.cfg.iterations):
            self._iteration = it
            t0 = time.time()

            # 1. Self-play
            games = self._run_self_play(self.cfg.games_per_iteration)
            for g in games:
                self.buffer.add_game(g)

            # 2. Update
            loss = None
            if self.buffer.num_games >= self.cfg.min_buffer_games:
                loss = self._update_step()
                self.loss_history.append(loss)

            # 3. Eval
            eval_result = None
            if (it + 1) % self.cfg.eval_every == 0:
                eval_result = self.evaluate(self.cfg.eval_games)
                self.win_history.append(eval_result)

            if self.cfg.verbose:
                elapsed = time.time() - t0
                loss_str = f"loss={loss:.4f}" if loss is not None else "loss=N/A (warming up)"
                eval_str = ""
                if eval_result:
                    eval_str = (f"  eval: X={eval_result['X_wins']:.0%} "
                                f"D={eval_result['draws']:.0%} O={eval_result['O_wins']:.0%}")
                print(f"[iter {it+1:3d}/{self.cfg.iterations}]  "
                      f"{loss_str}  {self.buffer}  {elapsed:.1f}s{eval_str}")

    # ── Self-play ─────────────────────────────────────────────────────────────
    def _run_self_play(self, n_games: int) -> List[GameRecord]:
        records = []
        for _ in range(n_games):
            records.append(self._play_one_game())
        return records

    def _play_one_game(self) -> GameRecord:
        env    = TicTacToeEnv()
        record = GameRecord()
        obs    = env.reset()
        done   = False
        step   = 0

        while not done:
            temp = (self.cfg.temperature_init
                    if step < self.cfg.temperature_threshold
                    else self.cfg.temperature_final)

            result: SearchResult = self.mcts.search(obs, env.legal_actions, temperature=temp)

            transition = Transition(
                observation=obs.copy(),
                action=result.action,
                reward=0.0,          # filled in after game ends
                policy_targets=result.policy_probs.copy(),
                value_target=0.0,    # filled in by compute_value_targets
                to_play=env.to_play,
                legal_actions=list(env.legal_actions),
                done=False,
            )

            obs, reward, done, info = env.step(result.action)

            # Patch reward into the transition
            transition = Transition(
                observation=transition.observation,
                action=transition.action,
                reward=reward,
                policy_targets=transition.policy_targets,
                value_target=transition.value_target,
                to_play=transition.to_play,
                legal_actions=transition.legal_actions,
                done=done,
            )
            record.add(transition)
            step += 1

        record.winner = info["winner"]
        record.compute_value_targets(discount=self.cfg.discount)
        return record

    # ── Gradient update ───────────────────────────────────────────────────────
    def _update_step(self) -> float:
        """
        One gradient step using SPSA (simultaneous perturbation stochastic approximation).

        For each perturbation pair (+Δ, -Δ):
          • compute loss with θ+εΔ and θ-εΔ
          • estimate gradient: ĝ ≈ (L+ - L-) / (2ε) · Δ⁻¹
        """
        cfg  = self.cfg
        batch = self.buffer.sample_unroll_batch(cfg.batch_size, cfg.unroll_steps)
        if not batch:
            return 0.0

        eps   = cfg.spsa_epsilon
        base_weights = self.model.weights.flat()
        n_params     = len(base_weights)
        grad_accum   = np.zeros(n_params, dtype=np.float64)

        for _ in range(cfg.spsa_samples):
            delta  = self.rng.choice([-1.0, 1.0], size=n_params).astype(np.float64)
            w_plus  = base_weights + eps * delta
            w_minus = base_weights - eps * delta

            l_plus  = self._batch_loss(batch, w_plus)
            l_minus = self._batch_loss(batch, w_minus)
            grad_accum += (l_plus - l_minus) / (2 * eps) / delta

        grad_accum /= cfg.spsa_samples

        # Reconstruct NetworkWeights from gradient vector
        grad_weights = self._vec_to_weights(grad_accum.astype(np.float32))

        # L2 weight decay
        for key in self.model.weights.__dict__:
            cur  = getattr(self.model.weights, key)
            grad = getattr(grad_weights, key)
            setattr(grad_weights, key, grad + cfg.weight_decay * cur)

        self.model.apply_gradient(grad_weights, lr=cfg.learning_rate)

        # Return average loss at current params
        return self._batch_loss(batch, base_weights)

    def _batch_loss(self, batch: List[List[Transition]], weight_vec: np.ndarray) -> float:
        """Evaluate total loss for a batch using a given flat weight vector."""
        # Temporarily swap weights
        original = self.model.weights.copy()
        self.model.weights = self._vec_to_weights(weight_vec.astype(np.float32))
        total_loss = 0.0
        count = 0

        for trajectory in batch:
            if not trajectory:
                continue
            first = trajectory[0]
            latent = self.model.represent(first.observation)

            for step_idx, tr in enumerate(trajectory):
                # Prediction loss
                pred = self.model.predict(latent)

                l_v = mse(pred.value, np.clip(tr.value_target, -1, 1))
                l_pi = cross_entropy(pred.policy_logits, tr.policy_targets)
                total_loss += (self.cfg.value_loss_weight  * l_v +
                               self.cfg.policy_loss_weight * l_pi)

                if step_idx < len(trajectory) - 1:
                    # Dynamics loss
                    dyn = self.model.step(latent, tr.action)
                    r_target = scalar_to_support(tr.reward, self.REWARD_SUPPORT)
                    l_r = cross_entropy(dyn.reward_logits, r_target)
                    total_loss += self.cfg.reward_loss_weight * l_r
                    latent = dyn.next_latent

                count += 1

        self.model.weights = original
        return total_loss / max(count, 1)

    def _vec_to_weights(self, vec: np.ndarray) -> NetworkWeights:
        """Reconstruct a NetworkWeights object from a flat parameter vector."""
        template = self.model.weights
        new_w    = {}
        offset   = 0
        for key, val in template.__dict__.items():
            size        = val.size
            new_w[key]  = vec[offset:offset+size].reshape(val.shape)
            offset     += size
        return NetworkWeights(**new_w)

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, n_games: int = 20) -> dict:
        """
        Play N games with temperature=0 (greedy) and report win rates.
        Uses a greedy random opponent for player O.
        """
        wins = {1: 0, -1: 0, 0: 0}

        for _ in range(n_games):
            env  = TicTacToeEnv()
            obs  = env.reset()
            done = False

            while not done:
                if env.to_play == PLAYER_X:
                    # MCTS agent (greedy)
                    result = self.mcts.search(obs, env.legal_actions, temperature=0.0)
                    action = result.action
                else:
                    # Random opponent
                    legal  = env.legal_actions
                    action = int(self.rng.choice(legal))

                obs, _, done, info = env.step(action)

            winner = info["winner"]
            wins[winner if winner is not None else 0] += 1

        total = n_games
        return {
            "X_wins": wins[1]  / total,
            "draws":  wins[0]  / total,
            "O_wins": wins[-1] / total,
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str = "checkpoint") -> None:
        self.model.save(path)

    def load_checkpoint(self, path: str = "checkpoint") -> None:
        loaded = WorldModel.load(path)
        self.model.weights = loaded.weights
        self.mcts = MCTS(self.model, MCTSConfig(num_simulations=self.cfg.mcts_simulations))

# Blind MCTS TicTacToe — MuZero Style

A clean, dependency-free Python implementation of **MuZero-style planning**
applied to TicTacToe. The agent plans entirely in **latent space** — after
encoding the board once with `h()`, it never queries the real environment again.

---

## Architecture

```
environment.py    — Real TicTacToe MDP (rules, state, transitions)
world_model.py    — Three MuZero networks (h, f, g) as NumPy MLPs
mcts.py           — Blind MCTS planner (PUCT, Dirichlet noise, backup)
replay_buffer.py  — Circular buffer of game trajectories
trainer.py        — Self-play → buffer → SPSA gradient update loop
main.py           — Entry point (train / play / ai_vs_ai / analyze / demo)
```

---

## The Three Networks

| Network | Signature | Role |
|---------|-----------|------|
| `h()` Representation | `obs (3×3×3) → z (16,)` | Encode real board into latent state |
| `f()` Prediction     | `z → policy (9,), value` | Prior + value estimate for MCTS |
| `g()` Dynamics       | `z, action → z', reward` | Transition in latent space (no env!) |

MCTS calls `h()` **once** at the root, then expands purely via `f()` and `g()`.

---

## MCTS Algorithm

```
for each simulation:
    SELECT   — follow PUCT (Q + exploration bonus) until leaf
    EXPAND   — call g(z, a) → z'   then   f(z') → π, v
    BACKUP   — propagate v up the path, flipping sign (zero-sum)

action = argmax visit_counts  (or sample with temperature)
```

**PUCT score:**
```
Q(s,a) + P(s,a) · √N(s) / (1+N(s,a)) · (c1 + log((N(s)+c2+1)/c2))
```

---

## Usage

```bash
# Quick demo (short training + one game + position analysis)
python main.py demo

# Full training (saves model.npz)
python main.py train

# Human vs AI (loads saved model)
python main.py play

# Watch AI vs AI (5 games)
python main.py ai_vs_ai 5

# Analyse a specific position ('.' = empty)
python main.py analyze "X.O....X."
```

---

## Training

The trainer uses **SPSA** (gradient-free) to update all three networks jointly:

1. Self-play N games with MCTS → store in `ReplayBuffer`
2. Sample unrolled sub-trajectories (length K)
3. Compute three losses per step:
   - **L_v** — MSE between `f(z).value` and bootstrapped return Gₜ
   - **L_π** — cross-entropy between `f(z).policy` and MCTS visit distribution
   - **L_r** — cross-entropy between `g(z,a).reward` and observed reward
4. SPSA gradient estimate → SGD update

---

## Dependencies

**None.** Only the Python standard library + NumPy.

```bash
pip install numpy
python main.py demo
```

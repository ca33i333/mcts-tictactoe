"""
main.py
───────
Entry point for the MuZero-style Blind MCTS TicTacToe system.

Modes
─────
  python main.py train          — run self-play training loop
  python main.py play           — human (X) vs trained AI (O)
  python main.py ai_vs_ai       — watch two MCTS agents play
  python main.py demo           — quick demo (short training + one game)
  python main.py analyze <pos>  — analyse a board position with MCTS
"""

from __future__ import annotations
import sys
import time
import numpy as np

from environment   import TicTacToeEnv, TicTacToeState, NUM_ACTIONS, PLAYER_X, PLAYER_O
from world_model   import WorldModel, NetworkWeights
from mcts          import MCTS, MCTSConfig, SearchResult
from replay_buffer import ReplayBuffer
from trainer       import Trainer, TrainConfig


# ── Rendering helpers ─────────────────────────────────────────────────────────
BORDER = "─" * 36

def clear() -> None:
    print("\033[2J\033[H", end="")

def print_board(state: TicTacToeState, highlight: list = None) -> None:
    sym  = {1: "\033[91mX\033[0m", -1: "\033[96mO\033[0m", 0: "·"}
    hl   = set(highlight or [])
    rows = []
    for r in range(3):
        parts = []
        for c in range(3):
            idx = r * 3 + c
            s   = sym[state.board[idx]]
            if idx in hl:
                s = f"\033[93m{s}\033[0m"
            parts.append(f" {s} ")
        rows.append("│".join(parts))
    print("┌───┬───┬───┐")
    for i, row in enumerate(rows):
        print(f"│{row}│")
        if i < 2:
            print("├───┼───┼───┤")
    print("└───┴───┴───┘")

def print_policy(result: SearchResult) -> None:
    visits = result.visit_counts
    total  = visits.sum() or 1
    qvals  = result.child_q_values
    print(f"\n  MCTS policy (visit%)   Q-values")
    for r in range(3):
        vrow = []
        qrow = []
        for c in range(3):
            i = r * 3 + c
            pct = visits[i] / total * 100
            vrow.append(f"{pct:5.1f}%")
            qrow.append(f"{qvals[i]:+.2f}" if visits[i] > 0 else "  —  ")
        print(f"  {'  '.join(vrow)}     {'  '.join(qrow)}")
    print(f"\n  Root Q={result.root_value:+.3f}   "
          f"depth={result.max_depth}   sims={result.total_sims}")


# ── Mode: Training ────────────────────────────────────────────────────────────
def run_training(checkpoint: str = "model") -> WorldModel:
    print(BORDER)
    print("  Blind MCTS · MuZero TicTacToe · Training")
    print(BORDER)

    config = TrainConfig(
        iterations=40,
        games_per_iteration=25,
        mcts_simulations=60,
        batch_size=64,
        learning_rate=0.008,
        eval_every=5,
        eval_games=30,
        verbose=True,
    )

    model   = WorldModel()
    trainer = Trainer(model, config)

    print(f"Networks: repr({27}→{16}), pred({16}→π,v), dyn({16+9}→{16},r)")
    print(f"Training: {config.iterations} iters × {config.games_per_iteration} games, "
          f"{config.mcts_simulations} MCTS sims\n")

    trainer.train()

    trainer.save_checkpoint(checkpoint)
    print(f"\nFinal evaluation ({config.eval_games} games):")
    eval_r = trainer.evaluate(config.eval_games)
    print(f"  X wins: {eval_r['X_wins']:.0%}   draws: {eval_r['draws']:.0%}   O wins: {eval_r['O_wins']:.0%}")
    return model


# ── Mode: Human vs AI ─────────────────────────────────────────────────────────
def run_play(model: WorldModel, num_sims: int = 200) -> None:
    mcts_cfg = MCTSConfig(num_simulations=num_sims)
    mcts     = MCTS(model, mcts_cfg)
    env      = TicTacToeEnv()

    scores = {1: 0, -1: 0, 0: 0}

    print(BORDER)
    print("  Human (X) vs Blind MCTS AI (O)")
    print(f"  AI uses {num_sims} MCTS simulations per move")
    print("  Enter cell number (0-8), layout:")
    print("   0 │ 1 │ 2")
    print("   3 │ 4 │ 5")
    print("   6 │ 7 │ 8")
    print(BORDER)

    while True:
        obs  = env.reset()
        done = False

        while not done:
            print_board(env.state)

            if env.to_play == PLAYER_X:
                # Human move
                legal = env.legal_actions
                while True:
                    try:
                        raw = input(f"\n  Your move {legal}: ").strip()
                        action = int(raw)
                        if action in legal:
                            break
                        print("  ✗ Illegal — try again.")
                    except (ValueError, EOFError):
                        print("  ✗ Enter a number.")
            else:
                # AI move
                print("\n  AI thinking…", end="", flush=True)
                t0 = time.time()
                result = mcts.search(obs, env.legal_actions, temperature=0.0)
                action = result.action
                elapsed = time.time() - t0
                print(f"\r  AI chose cell {action} in {elapsed:.2f}s")
                print_policy(result)

            obs, reward, done, info = env.step(action)

        # Game over
        print_board(env.state)
        winner = info["winner"]
        if winner == 1:
            print("\n  ╔══════════════╗")
            print("  ║   X  WINS!   ║")
            print("  ╚══════════════╝")
        elif winner == -1:
            print("\n  ╔══════════════╗")
            print("  ║   AI  WINS!  ║")
            print("  ╚══════════════╝")
        else:
            print("\n  ══ Draw! ══")

        scores[winner or 0] += 1
        print(f"  Score  X:{scores[1]}  Draw:{scores[0]}  O:{scores[-1]}\n")

        again = input("  Play again? [y/n]: ").strip().lower()
        if again != "y":
            break


# ── Mode: AI vs AI ────────────────────────────────────────────────────────────
def run_ai_vs_ai(model: WorldModel, n_games: int = 5, num_sims: int = 150) -> None:
    mcts_cfg = MCTSConfig(num_simulations=num_sims)
    mcts     = MCTS(model, mcts_cfg)
    scores   = {1: 0, -1: 0, 0: 0}

    print(BORDER)
    print(f"  AI vs AI  ({num_sims} sims per move)")
    print(BORDER)

    for game_idx in range(n_games):
        env  = TicTacToeEnv()
        obs  = env.reset()
        done = False

        print(f"\n  ── Game {game_idx+1} ──────────────────────")
        moves = []

        while not done:
            result = mcts.search(obs, env.legal_actions, temperature=0.1)
            action = result.action
            moves.append((env.to_play, action, result.root_value))
            obs, _, done, info = env.step(action)

        winner = info["winner"]
        scores[winner or 0] += 1

        print_board(env.state)
        print(f"  Moves: {[(('X' if p==1 else 'O'), a) for p,a,_ in moves]}")
        result_str = {1: "X wins", -1: "O wins", 0: "Draw"}[winner or 0]
        print(f"  Result: {result_str}")
        time.sleep(0.2)

    print(f"\n  Final: X={scores[1]}  Draw={scores[0]}  O={scores[-1]} / {n_games} games")


# ── Mode: Position analysis ───────────────────────────────────────────────────
def run_analyze(model: WorldModel, board_str: str = None, num_sims: int = 300) -> None:
    """
    Analyze a board position.
    board_str: 9 chars, e.g. 'X.O....X.' (X, O, or . for empty)
    """
    mcts_cfg = MCTSConfig(num_simulations=num_sims)
    mcts     = MCTS(model, mcts_cfg)

    if board_str and len(board_str) == 9:
        sym  = {"X": 1, "O": -1, ".": 0}
        board = tuple(sym.get(c.upper(), 0) for c in board_str)
        # Infer whose turn it is from piece count
        n_x = sum(1 for v in board if v == 1)
        n_o = sum(1 for v in board if v == -1)
        to_play = PLAYER_X if n_x == n_o else PLAYER_O
        state = TicTacToeState(board=board, to_play=to_play)
    else:
        state = TicTacToeState.initial()

    print(BORDER)
    print("  Position Analysis")
    print(BORDER)
    print_board(state)
    player_str = "X" if state.to_play == 1 else "O"
    print(f"  To play: {player_str}  |  Legal: {state.legal_actions()}")

    obs    = state.to_observation()
    legal  = state.legal_actions()
    if not legal:
        print("  Terminal position — no moves.")
        return

    result = mcts.search(obs, legal, temperature=0.0)
    print_policy(result)
    print(f"\n  Best move: cell {result.action}")

    # Highlight recommended move
    print("\n  Recommended:")
    print_board(state, highlight=[result.action])


# ── Mode: Quick demo ──────────────────────────────────────────────────────────
def run_demo() -> None:
    print(BORDER)
    print("  Blind MCTS TicTacToe — Quick Demo")
    print(BORDER)

    # Short training
    config = TrainConfig(
        iterations=10,
        games_per_iteration=10,
        mcts_simulations=30,
        batch_size=32,
        eval_every=5,
        eval_games=10,
        verbose=True,
    )
    model   = WorldModel()
    trainer = Trainer(model, config)
    trainer.train()

    # One AI vs AI game
    print("\n  Demo game (AI vs AI, 100 sims):")
    run_ai_vs_ai(model, n_games=1, num_sims=100)

    # Analyze starting position
    print("\n  Analysis of opening position:")
    run_analyze(model, num_sims=200)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]
    mode = args[0] if args else "demo"
    checkpoint = "model"

    if mode == "train":
        run_training(checkpoint)

    elif mode == "play":
        try:
            model = WorldModel.load(checkpoint)
            print("[main] Loaded saved model.")
        except FileNotFoundError:
            print("[main] No saved model found — training first…")
            model = run_training(checkpoint)
        run_play(model, num_sims=200)

    elif mode == "ai_vs_ai":
        try:
            model = WorldModel.load(checkpoint)
        except FileNotFoundError:
            print("[main] No saved model — using random weights.")
            model = WorldModel()
        n = int(args[1]) if len(args) > 1 else 5
        run_ai_vs_ai(model, n_games=n, num_sims=150)

    elif mode == "analyze":
        try:
            model = WorldModel.load(checkpoint)
        except FileNotFoundError:
            model = WorldModel()
        board_str = args[1] if len(args) > 1 else None
        run_analyze(model, board_str, num_sims=300)

    elif mode == "demo":
        run_demo()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [train|play|ai_vs_ai|demo|analyze [board]]")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Chess Reinforcement Learning — First-Visit Monte Carlo Control

A Google Colab project demonstrating how **two chess agents (White and Black) learn a move-selection policy through self-play** using **First-Visit Monte Carlo Control** and a **decaying ε-greedy policy**.

The aim is to demonstrate reinforcement learning concepts from *Chapter 5: Reinforcement Learning*, especially First-Visit Monte Carlo (page 39), Monte Carlo Control (pages 41–42), and ε-greedy action selection (page 48). This is an **educational experiment**, not a competitive chess engine.

## Project overview

| Component | Implementation |
|---|---|
| Environment | Chess with legal moves enforced by `python-chess` |
| Agents | Separate White and Black Monte Carlo agents |
| Learning algorithm | First-Visit Monte Carlo **Control** |
| Move-selection policy | Decaying **ε-greedy** during training; greedy (`ε = 0`) in evaluation |
| State | Abstract position features from the current player's perspective |
| Action | Category of a legal chess move |
| Reward | `+1` win, `0` draw, `−1` loss, from each player's perspective |
| Training | 800 self-play episodes by default |
| Evaluation | Each trained agent vs. a random legal-move opponent |

## How the strategy works

### 1. ε-greedy move selection

For a given chess state `s`, the agent chooses between:

- **Exploration**, with probability `ε`: select a random legal move.
- **Exploitation**, with probability `1 − ε`: select a legal move whose action category has the highest learned `Q(s, a)` value.

```text
Observe chess position
         |
         v
Draw a random number
         |
    +----+----+
    |         |
  < ε        ≥ ε
    |         |
Explore     Exploit
(random)   (highest Q)
    |         |
    +----+----+
         |
     Play move
         |
 Continue until episode ends
         |
   Receive final reward
         |
 First-Visit MC Q-value update
         |
  Next episode: improved policy
```

Exploration decreases gradually from `ε = 1.0` to `ε = 0.05` across the configured training episodes:

```python
def training_epsilon(index, total):
    fraction = index / max(1, total - 1)
    return EPS_START * (EPS_END / EPS_START) ** fraction
```

During demonstrations and evaluation, `ε = 0`, so the agents exploit their learned values. When several moves tie for the highest value, the notebook breaks ties randomly.

### 2. Reward function

Rewards are determined at the end of an episode:

| Outcome | White reward | Black reward |
|---|---:|---:|
| White wins | +1 | −1 |
| Black wins | −1 | +1 |
| Draw | 0 | 0 |

**Important:** Episodes stop after at most **120 half-moves** by default. If no official chess result occurs by then, the notebook uses a **material-based educational adjudication** (a lead of at least 2 pawn units produces a win; otherwise a draw). These capped results are **not official chess wins**.

### 3. First-Visit Monte Carlo Control

After each completed episode:

1. Collect each player's sequence of visited state–action pairs.
2. For each player, retain **only the first visit** to a given pair within that episode.
3. Compute the episode return `G` from the final reward (discount factor `γ = 1.0` by default).
4. Update the empirical action-value estimate:

\[
Q(s,a) = \frac{\sum_{i=1}^{N(s,a)} G_i}{N(s,a)}
\]

5. Use the updated Q-values to choose future moves with ε-greedy selection.

Unlike **Monte Carlo Prediction**, which evaluates a fixed policy, **Monte Carlo Control improves the action-selection policy**. Unlike **Q-Learning**, Monte Carlo learns from completed episodes instead of bootstrapped one-step updates.

## State and action representation

Full chess positions and exact moves make a tabular Q-table impractical with a small number of games. The notebook therefore groups similar positions and moves.

**State features:**

- Game phase (derived from the move count)
- Material advantage bucket, from the player-to-move perspective
- Whether the current player is in check
- Whether each player still has a queen

**Action categories** include pawn advances, center-pawn moves, minor-piece moves, rook/queen/king moves, captures grouped by captured-piece type, checks, castling, and promotions.

This generalization makes learning possible in a classroom-scale experiment, but it also discards chess information (such as tactical threats and precise piece locations). **Moves that share a category may receive the same learned Q-value.**

## Files

```text
.
├── README.md
└── First_Visit_Monte_Carlo_Control_Chess_Epsilon_Greedy.ipynb
```

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `First_Visit_Monte_Carlo_Control_Chess_Epsilon_Greedy.ipynb`.
3. Select **Runtime → Run all**.
4. The notebook installs the required packages, trains both agents, produces plots, evaluates them, and displays a learned-policy chess game step by step.

The notebook installs `python-chess`, `pymunk`, and `Pillow` in dedicated cells. It also uses `numpy`, `matplotlib`, `tqdm`, and IPython display facilities, typically available in Colab.

To run locally instead, install dependencies with:

```bash
pip install python-chess numpy matplotlib tqdm pymunk Pillow ipython jupyter
jupyter notebook First_Visit_Monte_Carlo_Control_Chess_Epsilon_Greedy.ipynb
```

## Configuration

Edit the configuration cell before running training:

```python
SEED = 42
TRAIN_EPISODES = 800
MAX_PLIES = 120
GAMMA = 1.0
EPS_START = 1.0
EPS_END = 0.05
EVAL_GAMES = 30
MATERIAL_WIN_MARGIN = 2.0
```

For a longer experiment, increase `TRAIN_EPISODES` to `2000` or `5000`. This can increase runtime and does **not** guarantee a strong chess player.

## What the notebook displays

- Training progress and counts of White wins, Black wins, and draws
- Number of official finishes versus material-based capped finishes
- ε decay chart and training outcome chart
- Learned Q-value summaries by move category
- **Trained vs. untrained** agent evaluation against a random baseline
- An illustrated chess game with the **chosen move, action category, exploitation label, Q-value, and prior observation count** printed at each step
- An optional inline GIF demonstrating piece motion with Pymunk (visualization only)

## Limitations

- The learner uses **abstract state/action categories**, not exact chess positions and moves.
- Only hundreds of games are played by default, far too few to learn strong chess play.
- At the episode cap, material-based outcomes are teaching approximations, not official game results.
- Results against a **random opponent** are not measures of strength against human players or standard chess engines.
- This notebook does **not** implement Minimax, Alpha-Beta Pruning, or Monte Carlo Tree Search.
- The optional Pymunk animation does not affect chess rules, policies, or rewards.

## Key takeaway

**Learning algorithm:** First-Visit Monte Carlo Control  
**Policy:** Decaying ε-greedy for both White and Black  
**Learning signal:** Terminal rewards (+1 / 0 / −1)  
**Goal:** Learn better action-category preferences through self-play, rather than selecting moves randomly forever.

---

**Course reference:** *Chapter 5 — Reinforcement Learning*, sections on First-Visit Monte Carlo, Monte Carlo Control, and ε-greedy policies (pages 39–48).

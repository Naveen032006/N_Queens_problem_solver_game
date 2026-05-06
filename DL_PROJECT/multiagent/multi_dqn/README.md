# Multi-Agent DQN Traffic Signal Control
## 2×2 grid · 4 intersections · 1 agent per intersection

---

## Grid layout

```
  J24(0,200)      J23(100,200)
      |                 |
  J19─J15(0,100)─E12─J16(100,100)─J22
         |                 |
        E15               E13
         |                 |
  J20─J18(0,0)  ─E14─J17(100,0) ─J21
         |                 |
  J25(0,-100)       J26(100,-100)
```

---

## Quick start

```bash
pip install torch numpy matplotlib

# Set SUMO_HOME first
python train.py                  # 300 episodes headless
python train.py --gui            # watch in SUMO-GUI
python train.py --resume         # continue from checkpoints/
python train.py --episodes 30    # quick test

python plot_results.py           # 4 PNG files in logs/
python evaluate.py               # DQN vs fixed-time table
```

---

## Action space (identical at every junction, 16 linkIndices per junction)

| Action | Label | Phase | State string       | Active links        | Dur |
|--------|-------|-------|--------------------|---------------------|-----|
| 0 | **NSG**  | 0 | `GGGgrrrrGGGgrrrr` | [0-3, 8-11]  N+S all | 35s |
| 1 | **NSLG** | 2 | `rrrGrrrrrrrGrrrr` | [3, 11]  N+S left   | 15s |
| 2 | **EWG**  | 4 | `rrrrGGGgrrrrGGGg` | [4-7, 12-15] E+W all | 35s |
| 3 | **EWLG** | 6 | `rrrrrrrGrrrrrrrG` | [7, 15]  E+W left   | 15s |

G = protected green, g = permissive (left in mixed phase), y = yellow, r = red.
Yellow phases (3 s): indices 1, 3, 5, 7.

---

## Network structure (2 lanes per arm)

```
linkIndex pattern — IDENTICAL at all 4 junctions:
  N arm:  [0] l0→right  [1] l0→straight  [2] l1→straight  [3] l1→left
  E arm:  [4] l0→right  [5] l0→straight  [6] l1→straight  [7] l1→left
  S arm:  [8] l0→right  [9] l0→straight [10] l1→straight [11] l1→left
  W arm: [12] l0→right [13] l0→straight [14] l1→straight [15] l1→left
```

---

## State vector (16 values per intersection)

```
[q_N0, q_N1, q_E0, q_E1, q_S0, q_S1, q_W0, q_W1,
 w_N0, w_N1, w_E0, w_E1, w_S0, w_S1, w_W0, w_W1]
 ←────── queue (÷20) ──────→  ←────── wait (÷300) ──────→
```

Per-lane state lets the agent distinguish left-turn lane (l1) congestion
from through-traffic (l0) congestion and respond with NSLG/EWLG.

---

## Per-junction incoming lanes

| Junction | N arm          | E arm          | S arm          | W arm          |
|----------|----------------|----------------|----------------|----------------|
| J15      | E21_0, E21_1   | -E12_0, -E12_1 | E15_0, E15_1   | E16_0, E16_1   |
| J16      | E20_0, E20_1   | E19_0, E19_1   | -E13_0, -E13_1 | E12_0, E12_1   |
| J17      | E13_0, E13_1   | E18_0, E18_1   | E23_0, E23_1   | -E14_0, -E14_1 |
| J18      | -E15_0, -E15_1 | E14_0, E14_1   | E22_0, E22_1   | E17_0, E17_1   |

---

## Outputs

```
logs/training_log.csv         — per-episode metrics, all 4 agents
logs/reward_curves.png        — reward per intersection (2x2 panel)
logs/queue_wait_curves.png    — queue + wait per intersection
logs/action_distribution.png  — NSG/NSLG/EWG/EWLG fractions over time
logs/network_reward.png       — sum of all 4 rewards
checkpoints/{tid}_best.pt     — best model per intersection
checkpoints/{tid}_latest.pt   — latest checkpoint per intersection
```

"""
plot_results.py  —  Training curves for all 4 intersections.

Usage:  python plot_results.py  [--window 15]
Outputs 4 PNG files in logs/.
"""

import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
LOG_CSV  = os.path.join(HERE, "logs", "training_log.csv")
OUT_DIR  = os.path.join(HERE, "logs")
TL_IDS   = ['J15', 'J16', 'J17', 'J18']
TL_COLORS= {'J15':'#2563eb','J16':'#16a34a','J17':'#d97706','J18':'#dc2626'}
ACT_LBLS = ["NSG","NSLG","EWG","EWLG"]
ACT_COLS = ["#2563eb","#16a34a","#d97706","#dc2626"]


def smooth(data, w):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode="valid")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=15)
    return p.parse_args()


def plot(args):
    if not os.path.exists(LOG_CSV):
        print(f"Log not found: {LOG_CSV}\nRun train.py first.")
        return

    d = {tid: {'reward':[],'avg_queue':[],'avg_wait':[],'epsilon':[],'loss':[],
               'act':[[],[],[],[]]}
         for tid in TL_IDS}
    eps = []

    with open(LOG_CSV) as f:
        for row in csv.DictReader(f):
            eps.append(int(row["episode"]))
            for tid in TL_IDS:
                d[tid]['reward'].append(float(row[f"{tid}_reward"]))
                d[tid]['avg_queue'].append(float(row[f"{tid}_avg_queue"]))
                d[tid]['avg_wait'].append(float(row[f"{tid}_avg_wait"]))
                d[tid]['epsilon'].append(float(row[f"{tid}_epsilon"]))
                d[tid]['loss'].append(float(row[f"{tid}_loss"]))
                for i, lbl in enumerate(ACT_LBLS):
                    d[tid]['act'][i].append(int(row.get(f"{tid}_{lbl}", 0)))

    w, n = args.window, len(eps)

    # Plot 1: Reward
    fig, axes = plt.subplots(2, 2, figsize=(14,8), sharex=True)
    fig.suptitle("Episode Reward per Intersection", fontweight="bold")
    for i, tid in enumerate(TL_IDS):
        ax = axes[i//2][i%2]
        y  = d[tid]['reward']
        c  = TL_COLORS[tid]
        ax.plot(eps, y, alpha=0.2, color=c, linewidth=0.8)
        if n >= w: ax.plot(eps[w-1:], smooth(y,w), color=c, linewidth=2)
        ax.set_title(tid); ax.set_ylabel("Reward"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"reward_curves.png"), dpi=150)
    print("Saved → logs/reward_curves.png")
    plt.close()

    # Plot 2: Queue & Wait
    fig, axes = plt.subplots(2, 2, figsize=(14,8), sharex=True)
    fig.suptitle("Avg Queue & Wait per Intersection", fontweight="bold")
    for i, tid in enumerate(TL_IDS):
        ax = axes[i//2][i%2]
        yq, yw = d[tid]['avg_queue'], d[tid]['avg_wait']
        ax.plot(eps, yq, alpha=0.2, color='#dc2626', linewidth=0.8)
        ax.plot(eps, yw, alpha=0.2, color='#2563eb', linewidth=0.8)
        if n >= w:
            ax.plot(eps[w-1:], smooth(yq,w), color='#dc2626', linewidth=2, label="Queue")
            ax.plot(eps[w-1:], smooth(yw,w), color='#2563eb', linewidth=2, label="Wait(s)")
        ax.set_title(tid); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"queue_wait_curves.png"), dpi=150)
    print("Saved → logs/queue_wait_curves.png")
    plt.close()

    # Plot 3: Action distribution
    fig, axes = plt.subplots(2, 2, figsize=(14,8), sharex=True)
    fig.suptitle("Action Distribution per Intersection", fontweight="bold")
    for i, tid in enumerate(TL_IDS):
        ax  = axes[i//2][i%2]
        tot = np.maximum(np.array([sum(d[tid]['act'][a][j] for a in range(4))
                                   for j in range(n)], dtype=float), 1)
        bot = np.zeros(n)
        for a, (lbl, col) in enumerate(zip(ACT_LBLS, ACT_COLS)):
            fracs = np.array(d[tid]['act'][a], dtype=float) / tot
            ax.fill_between(eps, bot, bot+fracs, alpha=0.75, color=col, label=lbl)
            bot += fracs
        ax.set_title(tid); ax.set_ylim(0,1); ax.legend(fontsize=7); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"action_distribution.png"), dpi=150)
    print("Saved → logs/action_distribution.png")
    plt.close()

    # Plot 4: Network-wide reward
    net = [sum(d[tid]['reward'][j] for tid in TL_IDS) for j in range(n)]
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(eps, net, alpha=0.2, color='#7c3aed', linewidth=0.8)
    if n >= w:
        ax.plot(eps[w-1:], smooth(net,w), color='#7c3aed', linewidth=2, label=f"smooth w={w}")
    ax.set_title("Network-wide total reward (J15+J16+J17+J18)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Sum of rewards")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"network_reward.png"), dpi=150)
    print("Saved → logs/network_reward.png")
    plt.close()


if __name__ == "__main__":
    plot(parse_args())

"""
evaluate.py  —  Compare trained DQN vs fixed-time baseline.

Usage:
  python evaluate.py
  python evaluate.py --episodes 5
  python evaluate.py --gui
"""

import argparse, os
import numpy as np
from multi_env   import MultiIntersectionEnv, TL_IDS
from dqn_agent   import DQNAgent
from intersection import STATE_SIZE, ACTION_SIZE, ACTIONS

HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--gui",      action="store_true")
    return p.parse_args()


def run_episode(env, agents=None, fixed_cycle=7):
    states  = env.reset()
    phase   = {tid: 0 for tid in TL_IDS}
    timer   = {tid: 0 for tid in TL_IDS}
    ah      = {tid: [0,0,0,0] for tid in TL_IDS}

    while True:
        if agents:
            actions = {tid: agents[tid].act(states[tid], training=False)
                       for tid in TL_IDS}
        else:
            actions = {}
            for tid in TL_IDS:
                actions[tid] = phase[tid]
                timer[tid]  += 1
                if timer[tid] >= fixed_cycle:
                    phase[tid] = (phase[tid] + 1) % 4
                    timer[tid] = 0
        for tid in TL_IDS:
            ah[tid][actions[tid]] += 1
        results = env.step(actions)
        for tid in TL_IDS:
            states[tid] = results[tid][0]
        if results[TL_IDS[0]][2]:
            break

    summary = env.get_episode_summary()
    steps   = max(1, env.sim_step)
    return {
        tid: {
            "reward":    summary[tid]["episode_reward"],
            "avg_queue": summary[tid]["total_queue"] / steps,
            "avg_wait":  summary[tid]["total_wait"]  / steps,
            "act_hist":  ah[tid],
        }
        for tid in TL_IDS
    }


def evaluate(args):
    agents = {}
    missing = []
    for tid in TL_IDS:
        path  = os.path.join(CKPT_DIR, f"{tid}_best.pt")
        agent = DQNAgent(tl_id=tid, state_size=STATE_SIZE, action_size=ACTION_SIZE)
        if os.path.exists(path):
            agent.load(path)
            agent.epsilon = 0.0
        else:
            missing.append(tid)
        agents[tid] = agent

    if missing:
        print(f"Missing checkpoints: {missing}")
        print("Run  python train.py  first.")
        return

    env = MultiIntersectionEnv(use_gui=args.gui)
    dqn_all, fix_all = [], []

    print(f"\nEvaluating {args.episodes} episodes ...\n")
    for ep in range(1, args.episodes + 1):
        d = run_episode(env, agents=agents)
        dqn_all.append(d)
        print(f"  [DQN ep {ep}]")
        for tid in TL_IDS:
            print(f"    {tid}  R={d[tid]['reward']:8.1f}  "
                  f"Q={d[tid]['avg_queue']:.2f}  W={d[tid]['avg_wait']:.1f}s")

        f = run_episode(env)   # fixed-time
        fix_all.append(f)
        print(f"  [FIX ep {ep}]")
        for tid in TL_IDS:
            print(f"    {tid}  R={f[tid]['reward']:8.1f}  "
                  f"Q={f[tid]['avg_queue']:.2f}  W={f[tid]['avg_wait']:.1f}s")
        print()

    env.close()

    # Summary
    print("=" * 62)
    print(f"  {'Junction':<8} {'DQN Reward':>12} {'Fix Reward':>12} "
          f"{'Q imp':>8} {'W imp':>8}")
    print("-" * 62)
    for tid in TL_IDS:
        dr = np.mean([r[tid]["reward"]    for r in dqn_all])
        fr = np.mean([r[tid]["reward"]    for r in fix_all])
        dq = np.mean([r[tid]["avg_queue"] for r in dqn_all])
        fq = np.mean([r[tid]["avg_queue"] for r in fix_all])
        dw = np.mean([r[tid]["avg_wait"]  for r in dqn_all])
        fw = np.mean([r[tid]["avg_wait"]  for r in fix_all])
        qi = (fq - dq) / max(1e-9, fq) * 100
        wi = (fw - dw) / max(1e-9, fw) * 100
        print(f"  {tid:<8} {dr:>12.1f} {fr:>12.1f} {qi:>+7.1f}% {wi:>+7.1f}%")

    net_dqn = np.mean([sum(r[t]["reward"] for t in TL_IDS) for r in dqn_all])
    net_fix = np.mean([sum(r[t]["reward"] for t in TL_IDS) for r in fix_all])
    print("-" * 62)
    print(f"  {'TOTAL':<8} {net_dqn:>12.1f} {net_fix:>12.1f}")
    print("=" * 62)


if __name__ == "__main__":
    evaluate(parse_args())
"""
train.py  —  Train 4 independent DQN agents (one per intersection).

All 4 agents act simultaneously each step.
Each agent has its own local state (16 values), local reward, and replay buffer.

Usage:
  python train.py                 headless, 300 episodes
  python train.py --gui           SUMO-GUI (slower)
  python train.py --resume        continue from checkpoints/
  python train.py --episodes 30   quick test
"""

import argparse, csv, os, time
from multi_env   import MultiIntersectionEnv, TL_IDS
from dqn_agent   import DQNAgent
from intersection import STATE_SIZE, ACTION_SIZE, ACTIONS

CFG = dict(
    episodes        = 500,
    episode_length  = 3600,
    gamma           = 0.99,
    lr              = 1e-3,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    epsilon_decay   = 0.992,
    memory_size     = 50_000,
    batch_size      = 64,
    target_update   = 10,
    save_every      = 10,
    warmup_steps    = 500,
)

HERE     = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(HERE, "logs")
CKPT_DIR = os.path.join(HERE, "checkpoints")


def ckpt(tid, tag="latest"):
    return os.path.join(CKPT_DIR, f"{tid}_{tag}.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gui",      action="store_true")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--episodes", type=int, default=None)
    return p.parse_args()


def train(args):
    os.makedirs(LOG_DIR,  exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    n_ep = args.episodes or CFG["episodes"]

    env    = MultiIntersectionEnv(use_gui=args.gui,
                                   episode_length=CFG["episode_length"])
    agents = {
        tid: DQNAgent(
            tl_id=tid, state_size=STATE_SIZE, action_size=ACTION_SIZE,
            lr=CFG["lr"], gamma=CFG["gamma"],
            epsilon_start=CFG["epsilon_start"],
            epsilon_end=CFG["epsilon_end"],
            epsilon_decay=CFG["epsilon_decay"],
            memory_size=CFG["memory_size"],
            batch_size=CFG["batch_size"],
        )
        for tid in TL_IDS
    }

    start_ep    = 1
    best_reward = {tid: float("-inf") for tid in TL_IDS}
    global_step = 0

    if args.resume:
        for tid, agent in agents.items():
            p = ckpt(tid)
            if os.path.exists(p):
                agent.load(p)
        log_csv = os.path.join(LOG_DIR, "training_log.csv")
        if os.path.exists(log_csv):
            with open(log_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                start_ep = int(rows[-1]["episode"]) + 1
                for tid in TL_IDS:
                    best_reward[tid] = max(float(r[f"{tid}_reward"]) for r in rows)
        print(f"Resuming from episode {start_ep}")

    log_csv  = os.path.join(LOG_DIR, "training_log.csv")
    csv_file = open(log_csv, "a", newline="")
    writer   = csv.writer(csv_file)
    if start_ep == 1:
        header = ["episode", "duration_s", "sim_steps"]
        for tid in TL_IDS:
            header += [f"{tid}_reward", f"{tid}_avg_queue", f"{tid}_avg_wait",
                       f"{tid}_epsilon", f"{tid}_loss",
                       f"{tid}_NSG", f"{tid}_NSLG", f"{tid}_EWG", f"{tid}_EWLG"]
        writer.writerow(header)

    print("=" * 72)
    print("  Multi-Agent DQN  —  2x2 grid  —  4 intersections x 4 actions")
    print()
    print("  Grid:  J15(top-left)   J16(top-right)")
    print("         J18(bot-left)   J17(bot-right)")
    print()
    print("  Action 0  NSG   N+S all moves (left=permissive)")
    print("  Action 1  NSLG  N+S left protected")
    print("  Action 2  EWG   E+W all moves (left=permissive)")
    print("  Action 3  EWLG  E+W left protected")
    print()
    print(f"  Episodes={n_ep}  Length={CFG['episode_length']}s  "
          f"State={STATE_SIZE}  Actions={ACTION_SIZE}  "
          f"Net=16->256->256->4")
    print("=" * 72)

    for ep in range(start_ep, start_ep + n_ep):
        t0       = time.time()
        states   = env.reset()
        losses   = {tid: [] for tid in TL_IDS}
        act_hist = {tid: [0,0,0,0] for tid in TL_IDS}

        while True:
            actions = {tid: agents[tid].act(states[tid], training=True)
                       for tid in TL_IDS}
            for tid in TL_IDS:
                act_hist[tid][actions[tid]] += 1

            results = env.step(actions)
            global_step += 1

            any_done = False
            for tid in TL_IDS:
                ns, reward, done, info = results[tid]
                agents[tid].remember(states[tid], actions[tid], reward, ns, done)
                states[tid] = ns
                if done: any_done = True
                if global_step >= CFG["warmup_steps"]:
                    loss = agents[tid].replay()
                    if loss is not None:
                        losses[tid].append(loss)

            if any_done:
                break

        # End of episode
        for agent in agents.values():
            agent.decay_epsilon()
        if ep % CFG["target_update"] == 0:
            for agent in agents.values():
                agent.update_target_network()

        summary  = env.get_episode_summary()
        duration = time.time() - t0
        steps    = max(1, env.sim_step)

        for tid, agent in agents.items():
            ep_reward = summary[tid]["episode_reward"]
            # Only save best if agent is past exploration phase (ε < 0.3)
            if ep_reward > best_reward[tid] and agents[tid].epsilon < 0.3:
                best_reward[tid] = ep_reward
                agent.save(ckpt(tid, "best"))   
        if ep % CFG["save_every"] == 0:
            for tid, agent in agents.items():
                agent.save(ckpt(tid))
                agent.save(ckpt(tid, f"ep{ep:04d}"))

        # CSV row
        row = [ep, f"{duration:.1f}", steps]
        for tid in TL_IDS:
            s    = summary[tid]
            loss = sum(losses[tid]) / len(losses[tid]) if losses[tid] else 0.0
            row += [f"{s['episode_reward']:.2f}",
                    f"{s['total_queue']/steps:.3f}",
                    f"{s['total_wait']/steps:.1f}",
                    f"{agents[tid].epsilon:.4f}",
                    f"{loss:.5f}",
                    *act_hist[tid]]
        writer.writerow(row)
        csv_file.flush()

        # Console output
        print(f"\nEp {ep:4d}/{start_ep+n_ep-1}  [{duration:.0f}s]")
        for tid in TL_IDS:
            s    = summary[tid]
            loss = sum(losses[tid]) / len(losses[tid]) if losses[tid] else 0.0
            ah   = act_hist[tid]
            tot  = max(1, sum(ah))
            adist = " ".join(f"{ACTIONS[i][2]}:{100*ah[i]//tot:2d}%" for i in range(4))
            print(f"  {tid}  R={s['episode_reward']:8.1f}  "
                  f"Q={s['total_queue']/steps:.2f}  "
                  f"W={s['total_wait']/steps:.1f}s  "
                  f"ε={agents[tid].epsilon:.3f}  "
                  f"loss={loss:.4f}  {adist}")

    csv_file.close()
    print("\nTraining complete.")
    for tid in TL_IDS:
        print(f"  {tid} best reward: {best_reward[tid]:.2f}")


if __name__ == "__main__":
    train(parse_args())

"""
multi_env.py  —  Multi-agent SUMO environment.
One DQNAgent per junction (J15, J16, J17, J18), shared TraCI session.

Usage:
    env    = MultiIntersectionEnv()
    states = env.reset()                     # {tl_id: ndarray(16,)}
    results = env.step({tl_id: action, ...}) # {tl_id: (state,reward,done,info)}
"""

import os, sys
import numpy as np
from intersection import (
    Intersection, ACTIONS, YELLOW_DURATION, DECISION_STEP,
    STATE_SIZE, ACTION_SIZE
)

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "\nSUMO_HOME not set.\n"
        "Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo\n"
        "Linux:   export SUMO_HOME=/usr/share/sumo\n"
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

HERE     = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = os.path.join(HERE, "sumo_files", "multi1.sumocfg")
TL_IDS   = ['J15', 'J16', 'J17', 'J18']


class MultiIntersectionEnv:
    TL_IDS      = TL_IDS
    STATE_SIZE  = STATE_SIZE    # 16 per intersection
    ACTION_SIZE = ACTION_SIZE   # 4

    def __init__(self, use_gui: bool = False, episode_length: int = 3600):
        self.use_gui        = use_gui
        self.episode_length = episode_length
        self._running       = False
        self.sim_step       = 0
        self.intersections  = {tid: Intersection(tid) for tid in TL_IDS}

    def reset(self) -> dict:
        """Start new episode. Returns {tl_id: state_array (16,)}."""
        if self._running:
            try: traci.close()
            except: pass
            self._running = False

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([
            binary, "-c", SUMO_CFG,
            "--seed", "42", 
            "--random",
            "--no-step-log",         "true",
            "--waiting-time-memory", "3600",
            "--no-warnings",         "true",
            "--time-to-teleport",    "-1",
        ])
        self._running = True
        self.sim_step = 0

        for isc in self.intersections.values():
            isc._reset()
            isc.activate_initial_phase(traci)

        self._advance(10)   # warm-up: let vehicles spawn

        return {tid: isc.get_state(traci)
                for tid, isc in self.intersections.items()}

    def step(self, actions: dict) -> dict:
        """
        Simultaneous step for all agents.

        Parameters
        ----------
        actions : {tl_id: int}   each action ∈ {0,1,2,3}

        Returns
        -------
        {tl_id: (next_state, reward, done, info)}
        """
        # 1. Each intersection decides whether to switch
        switching = set()
        for tid, isc in self.intersections.items():
            if isc.decide_switch(actions[tid], traci):
                switching.add(tid)

        # 2. Yellow clearance if any intersection is switching
        if switching:
            self._advance(YELLOW_DURATION)
            for tid in switching:
                self.intersections[tid].complete_switch(traci)
            remaining = DECISION_STEP - YELLOW_DURATION
        else:
            remaining = DECISION_STEP

        # 3. Advance rest of decision period
        self._advance(remaining)

        done = self.sim_step >= self.episode_length

        # 4. Collect results
        results = {}
        for tid, isc in self.intersections.items():
            reward, arm_info = isc.tick(traci)
            next_state       = isc.get_state(traci)
            _, _, label, desc = ACTIONS[isc.current_action]
            info = {
                "sim_step":       self.sim_step,
                "green_timer":    isc.green_timer,
                "action":         isc.current_action,
                "phase_label":    label,
                "phase_desc":     desc,
                "total_queue":    isc.total_queue,
                "total_wait":     isc.total_wait,
                "episode_reward": isc.episode_reward,
                **arm_info,
            }
            results[tid] = (next_state, reward, done, info)

        if done:
            self.close()

        return results

    def get_episode_summary(self) -> dict:
        return {
            tid: {
                "total_queue":    isc.total_queue,
                "total_wait":     isc.total_wait,
                "episode_reward": isc.episode_reward,
            }
            for tid, isc in self.intersections.items()
        }

    def close(self):
        if self._running:
            try: traci.close()
            except: pass
            self._running = False

    def _advance(self, seconds: int):
        for _ in range(max(0, seconds)):
            traci.simulationStep()
            self.sim_step += 1

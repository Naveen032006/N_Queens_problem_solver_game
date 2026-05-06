"""
intersection.py  —  Per-intersection constants and logic.

Network: multi1.net.xml  (2x2 grid, 4 TL junctions)
Each junction: 4 arms x 2 lanes = 8 lanes, 16 linkIndices.

linkIndex pattern (IDENTICAL at every junction):
  N arm:  [0] l0-right  [1] l0-straight  [2] l1-straight  [3] l1-left
  E arm:  [4] l0-right  [5] l0-straight  [6] l1-straight  [7] l1-left
  S arm:  [8] l0-right  [9] l0-straight [10] l1-straight [11] l1-left
  W arm: [12] l0-right [13] l0-straight [14] l1-straight [15] l1-left

Action space (4 actions -> 8 phases):
  0  NSG   phase 0  GGGgrrrrGGGgrrrr  N+S all moves (left=permissive)
  1  NSLG  phase 2  rrrGrrrrrrrGrrrr  N+S left protected
  2  EWG   phase 4  rrrrGGGgrrrrGGGg  E+W all moves (left=permissive)
  3  EWLG  phase 6  rrrrrrrGrrrrrrrG  E+W left protected

State: 16 values (8 lane queues + 8 lane waits, normalised)
  order: N_l0, N_l1, E_l0, E_l1, S_l0, S_l1, W_l0, W_l1
"""

import numpy as np

# Action -> (green_phase_idx, yellow_phase_idx, label, description)
ACTIONS = {
    0: (0, 1, "NSG",  "N+S all moves — left permissive  [0-3,8-11]"),
    1: (2, 3, "NSLG", "N+S left protected               [3,11]"),
    2: (4, 5, "EWG",  "E+W all moves — left permissive  [4-7,12-15]"),
    3: (6, 7, "EWLG", "E+W left protected               [7,15]"),
}

STATE_SIZE  = 16
ACTION_SIZE = 4

YELLOW_DURATION = 3
MIN_GREEN       = 10
MAX_GREEN       = 60
DECISION_STEP   = 5

MAX_QUEUE = 20.0
MAX_WAIT  = 300.0

# Per-junction incoming lanes — order: N_l0, N_l1, E_l0, E_l1, S_l0, S_l1, W_l0, W_l1
JUNCTION_LANES = {
    'J15': ['E21_0',  'E21_1',  '-E12_0', '-E12_1', 'E15_0',  'E15_1',  'E16_0',  'E16_1'],
    'J16': ['E20_0',  'E20_1',  'E19_0',  'E19_1',  '-E13_0', '-E13_1', 'E12_0',  'E12_1'],
    'J17': ['E13_0',  'E13_1',  'E18_0',  'E18_1',  'E23_0',  'E23_1',  '-E14_0', '-E14_1'],
    'J18': ['-E15_0', '-E15_1', 'E14_0',  'E14_1',  'E22_0',  'E22_1',  'E17_0',  'E17_1'],
}

LANE_NAMES = ['N_l0','N_l1','E_l0','E_l1','S_l0','S_l1','W_l0','W_l1']
ARM_NAMES  = ['North', 'East', 'South', 'West']
ARM_LANE_IDX = {'North':[0,1], 'East':[2,3], 'South':[4,5], 'West':[6,7]}


class Intersection:
    """
    State, action logic, and metrics for one TL junction.
    TraCI connection is passed in from the multi-agent env.
    """

    def __init__(self, tl_id: str):
        self.tl_id  = tl_id
        self.lanes  = JUNCTION_LANES[tl_id]
        self._reset()

    def _reset(self):
        self.current_action  = 2      # start on EWG
        self.green_timer     = 0
        self.in_yellow       = False
        self._pending_target = None
        self.total_queue     = 0.0
        self.total_wait      = 0.0
        self.episode_reward  = 0.0

    def _q(self, lane, traci):
        try:    return traci.lane.getLastStepHaltingNumber(lane)
        except: return 0

    def _w(self, lane, traci):
        try:    return traci.lane.getWaitingTime(lane)
        except: return 0.0

    def get_state(self, traci) -> np.ndarray:
        """
        16-element float32 state vector.
        [0-7]  normalised queue per lane (N_l0,N_l1, E_l0,E_l1, S_l0,S_l1, W_l0,W_l1)
        [8-15] normalised wait  per lane (same order)
        Per-lane state lets agent distinguish left-turn vs through-traffic queues.
        """
        queues = np.array([self._q(l, traci) for l in self.lanes], dtype=np.float32)
        waits  = np.array([self._w(l, traci) for l in self.lanes], dtype=np.float32)
        return np.concatenate([
            np.clip(queues / MAX_QUEUE, 0.0, 1.0),
            np.clip(waits  / MAX_WAIT,  0.0, 1.0),
        ])

    def get_reward(self, traci) -> float:
        q = sum(self._q(l, traci) for l in self.lanes)
        w = sum(self._w(l, traci) for l in self.lanes)
        return float(-(q + 0.01 * w)/100)

    def decide_switch(self, action: int, traci) -> bool:
        """
        Evaluate switch guards, begin yellow if switching.
        Returns True if yellow phase was started.
        """
        if self.in_yellow:
            return False
        want_switch  = (action != self.current_action) and (self.green_timer >= MIN_GREEN)
        force_switch = (self.green_timer >= MAX_GREEN)
        if want_switch or force_switch:
            target = action if want_switch else (self.current_action + 1) % ACTION_SIZE
            self._pending_target = target
            _, yellow_idx, _, _ = ACTIONS[self.current_action]
            self.in_yellow = True
            traci.trafficlight.setPhase(self.tl_id, yellow_idx)
            return True
        return False

    def complete_switch(self, traci):
        """Activate green for pending target after yellow has elapsed."""
        green_idx, _, _, _ = ACTIONS[self._pending_target]
        traci.trafficlight.setPhase(self.tl_id, green_idx)
        self.current_action  = self._pending_target
        self.green_timer     = 0
        self.in_yellow       = False
        self._pending_target = None

    def activate_initial_phase(self, traci):
        green_idx, _, _, _ = ACTIONS[self.current_action]
        traci.trafficlight.setPhase(self.tl_id, green_idx)

    def tick(self, traci):
        """Update timers and accumulators. Returns (reward, arm_info)."""
        self.green_timer += DECISION_STEP
        reward = self.get_reward(traci)
        q = sum(self._q(l, traci) for l in self.lanes)
        w = sum(self._w(l, traci) for l in self.lanes)
        self.total_queue    += q
        self.total_wait     += w
        self.episode_reward += reward
        arm_info = {
            'queue_by_arm': {
                arm: sum(self._q(self.lanes[i], traci) for i in idxs)
                for arm, idxs in ARM_LANE_IDX.items()
            },
            'wait_by_arm': {
                arm: sum(self._w(self.lanes[i], traci) for i in idxs)
                for arm, idxs in ARM_LANE_IDX.items()
            },
        }
        return reward, arm_info

    def action_name(self, action: int) -> str:
        return ACTIONS[action][2]

    @property
    def phase_label(self) -> str:
        return ACTIONS[self.current_action][2]

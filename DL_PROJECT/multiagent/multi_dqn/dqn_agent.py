"""
dqn_agent.py  —  Double DQN with experience replay.
Network: 16 -> 256 -> 256 -> 4   (wider because state=16)
One instance per intersection.
"""

import random
import numpy as np
from collections import deque
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, action_size),
        )
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, tl_id, state_size=16, action_size=4,
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 memory_size=50_000, batch_size=64):
        self.tl_id         = tl_id
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = deque(maxlen=memory_size)

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, float(done)))

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(t).argmax().item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.FloatTensor(np.array(s)).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(np.array(s2)).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        cq = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            ba = self.policy_net(s2).argmax(1, keepdim=True)
            tq = r + self.gamma * self.target_net(s2).gather(1, ba).squeeze(1) * (1 - d)
        loss = F.smooth_l1_loss(cq, tq)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({'policy': self.policy_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optim':  self.optimizer.state_dict(),
                    'epsilon': self.epsilon, 'tl_id': self.tl_id}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck['policy'])
        self.target_net.load_state_dict(ck['target'])
        self.optimizer.load_state_dict(ck['optim'])
        self.epsilon = ck.get('epsilon', self.epsilon_end)
        print(f"  [{self.tl_id}] loaded {path}  ε={self.epsilon:.4f}")

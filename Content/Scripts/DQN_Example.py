# train.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from net_io import JsonLineClient

HOST, PORT = "127.0.0.1", 7777


class DQN(nn.Module):
    def __init__(self, obs_dim=4, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap=100000):
        self.b = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.b.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.b, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.b)


def select_action(qnet, obs, eps, n_actions=4):
    if random.random() < eps:
        a_r = random.randrange(n_actions)
        return a_r
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a_m = int(torch.argmax(qnet(x), dim=1).item())
        return a_m


def train_step(qnet, tgt, opt, replay, gamma=0.99, batch=64):
    s, a, r, s2, d = replay.sample(batch)
    s = torch.tensor(s, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
    r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    s2 = torch.tensor(s2, dtype=torch.float32)
    d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    q_sa = qnet(s).gather(1, a)
    with torch.no_grad():
        y = r + gamma * (1.0 - d) * tgt(s2).max(dim=1, keepdim=True)[0]

    loss = (q_sa - y).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def main():
    client = JsonLineClient(HOST, PORT)
    client.connect()

    qnet = DQN()
    tgt = DQN()
    tgt.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=1e-3)
    D = Replay() # replay buffer

    eps, eps_min, eps_decay = 1.0, 0.05, 0.9995 # for greedy policy
    train_after = 512 #2000
    train_every = 4
    C = 1000 # How often the target network params are copied 
    step = 0

    last_obs = None
    last_action = None

    for msg in client.iter_messages():
        typ = msg.get("type")

        if typ == "reset":
            obs = np.array(msg["obs"], dtype=np.float32)
            a = select_action(qnet, obs, eps)
            client.send({"type": "action", "a": a})
            last_obs, last_action = obs, a
            continue

        if typ == "step":
            obs2 = np.array(msg["obs"], dtype=np.float32)
            r = float(msg["reward"])
            done = 1 if msg["done"] else 0

            if last_obs is not None and last_action is not None:
                D.push(last_obs, last_action, r, obs2, done)

            a2 = select_action(qnet, obs2, eps)
            client.send({"type": "action", "a": a2})
            last_obs, last_action = obs2, a2

            step += 1
            eps = max(eps_min, eps * eps_decay)

            if len(D) >= train_after and step % train_every == 0:
                train_step(qnet, tgt, opt, D)

            if step % C == 0:
                tgt.load_state_dict(qnet.state_dict())
                print(f"step={step} eps={eps:.3f} replay={len(D)}")

            if step % 5000 == 0:
                torch.save(qnet.state_dict(), "dqn.pt")
                print("Save model")

            continue

        print("Unknown:", msg)


if __name__ == "__main__":
    main()
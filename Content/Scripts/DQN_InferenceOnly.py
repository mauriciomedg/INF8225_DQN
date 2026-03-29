import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
if not os.path.exists("dqn.pt"):
    raise FileNotFoundError("dqn.pt not found. Train first or copy weights here.")

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

def select_action(qnet, obs):
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a_m = int(torch.argmax(qnet(x), dim=1).item())
        return a_m

def main():
    client = JsonLineClient(HOST, PORT)
    client.connect()

    qnet = DQN()
            
    qnet.load_state_dict(torch.load("dqn.pt", map_location="cpu"))
    qnet.eval()

    for msg in client.iter_messages():
        typ = msg.get("type")

        if typ == "reset":
            obs = np.array(msg["obs"], dtype=np.float32)
            a = select_action(qnet, obs)
            client.send({"type": "action", "a": a})
            
            continue

        if typ == "step":
            if msg.get("done", False):
                client.send({"type": "action", "a": 0})
                continue

            obs2 = np.array(msg["obs"], dtype=np.float32)
            a2 = select_action(qnet, obs2)
            client.send({"type": "action", "a": a2})
            continue

        print("Unknown:", msg)


if __name__ == "__main__":
    main()
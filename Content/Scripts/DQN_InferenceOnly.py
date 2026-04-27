import numpy as np
import torch
import torch.nn as nn
import os

from net_io import JsonLineClient

HOST, PORT = "127.0.0.1", 7777
MODEL_PATH = "ddpg_actor_best.pt"   # or "ddpg_actor_latest.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train first or copy weights here.")


class Actor(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def select_action(actor, obs):
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = actor(x).squeeze(0).cpu().numpy()
        a = np.clip(a, -1.0, 1.0)
        return a


def main():
    client = JsonLineClient(HOST, PORT)
    client.connect()

    actor = Actor()
    actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    actor.eval()

    for msg in client.iter_messages():
        typ = msg.get("type")

        if typ == "reset":
            obs = np.array(msg["obs"], dtype=np.float32)
            a = select_action(actor, obs)
            client.send({
                "type": "action",
                "ax": float(a[0]),
                "ay": float(a[1]),
            })
            continue

        if typ == "step":
            if msg.get("done", False):
                continue

            obs2 = np.array(msg["obs"], dtype=np.float32)
            a2 = select_action(actor, obs2)
            client.send({
                "type": "action",
                "ax": float(a2[0]),
                "ay": float(a2[1]),
            })
            continue

        print("Unknown:", msg)


if __name__ == "__main__":
    main()
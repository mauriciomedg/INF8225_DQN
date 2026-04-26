import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import sys

from net_io import JsonLineClient

HOST, PORT = "127.0.0.1", 7777

LOG_PATH = "training_log.csv"

# Must match Unreal value
MAX_DISTANCE_METERS = 1.0

# Seuil de succès : un épisode est considéré réussi si la distance finale
# au joueur est inférieure à cette valeur. Choix arbitraire — à adapter
# selon l'échelle réelle de l'environnement Unreal.
SUCCESS_DISTANCE_M = 0.2

# Seed par défaut, surchargeable via argv[1].
DEFAULT_SEED = 42


def set_global_seed(seed: int) -> None:
    """Fixe les générateurs aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: le déterminisme complet de PyTorch nécessiterait aussi
    # torch.use_deterministic_algorithms(True), mais ça impacte les perfs
    # et ne change rien aux résultats côté CPU pour ce projet.

def append_training_log(row: dict):
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_reward",
                "episode_length",
                "final_distance",
                "success",
                "epsilon",
                "avg_loss",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


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
        return random.randrange(n_actions)

    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(qnet(x), dim=1).item())


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


def estimate_distance_from_obs(obs, max_distance_m):
    dx_n = float(obs[0])
    dy_n = float(obs[1])
    return (dx_n * dx_n + dy_n * dy_n) ** 0.5 * max_distance_m


def main():
    # Seed : argv[1] si fourni, sinon DEFAULT_SEED
    seed = DEFAULT_SEED
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"[WARN] seed invalide '{sys.argv[1]}', utilisation de {DEFAULT_SEED}")

    set_global_seed(seed)
    print(f"[INFO] seed = {seed}")												  
    client = JsonLineClient(HOST, PORT)
    client.connect()

    qnet = DQN()
    tgt = DQN()
    tgt.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=1e-3)
    D = Replay() # replay buffer

    eps, eps_min, eps_decay = 1.0, 0.05, 0.9995 # for greedy policy
    train_after = 512
    train_every = 4
    C = 1000 # How often the target network params are copied 
    step = 0

    last_obs = None
    last_action = None

    # episode stats
    episode_idx = 0
    ep_reward = 0.0
    ep_len = 0
    loss_sum = 0.0
    loss_count = 0
    final_distance = 0.0

    for msg in client.iter_messages():
        typ = msg.get("type")

        if typ == "reset":
            ep_reward = 0.0
            ep_len = 0
            loss_sum = 0.0
            loss_count = 0

            obs = np.array(msg["obs"], dtype=np.float32)
            final_distance = estimate_distance_from_obs(obs, MAX_DISTANCE_METERS)

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

            ep_reward += r
            ep_len += 1
            final_distance = estimate_distance_from_obs(obs2, MAX_DISTANCE_METERS)

            step += 1
            eps = max(eps_min, eps * eps_decay)

            if len(D) >= train_after and step % train_every == 0:
                loss_val = train_step(qnet, tgt, opt, D)
                loss_sum += loss_val
                loss_count += 1

            if step % C == 0:
                tgt.load_state_dict(qnet.state_dict())
                print(f"step={step} eps={eps:.3f} replay={len(D)}")

            if step % 5000 == 0:
                torch.save(qnet.state_dict(), "dqn.pt")
                print("Save model")

            if done:
                avg_loss = loss_sum / max(loss_count, 1)
                # success : 1 si l'agent a fini suffisamment près du joueur
                success = 1 if final_distance < SUCCESS_DISTANCE_M else 0																			
                append_training_log({
                    "episode": episode_idx,
                    "episode_reward": ep_reward,
                    "episode_length": ep_len,
                    "final_distance": final_distance,
                    "success": success,
                    "epsilon": eps,
                    "avg_loss": avg_loss,
                })

                print(
                    f"episode={episode_idx} "
                    f"reward={ep_reward:.3f} "
                    f"len={ep_len} "
                    f"final_dist={final_distance:.3f} "
                    f"success={success} " 
                    f"eps={eps:.3f} "
                    f"avg_loss={avg_loss:.6f}"
                )

                episode_idx += 1

            continue

        print("Unknown:", msg)


if __name__ == "__main__":
    main()
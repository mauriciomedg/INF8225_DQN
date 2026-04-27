import random
from collections import deque
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from net_io import JsonLineClient


HOST, PORT = "127.0.0.1", 7777
LOG_PATH = "training_log_ddpg.csv"

# Must match Unreal
MAX_DISTANCE_METERS = 1.0

DEFAULT_SEED = 42
DEVICE = torch.device("cpu")

OBS_DIM = 4
ACT_DIM = 2

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005

REPLAY_CAPACITY = 50000
BATCH_SIZE = 64
TRAIN_AFTER = 512
TRAIN_EVERY = 4

NOISE_STD = 0.1
NOISE_STD_MIN = 0.05
NOISE_DECAY = 0.9999


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                "noise_std",
                "avg_actor_loss",
                "avg_critic_loss",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),   # outputs in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)


class Replay:
    def __init__(self, cap=REPLAY_CAPACITY):
        self.b = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.b.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.b, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.b)


def estimate_distance_from_obs(obs, max_distance_m):
    dx_n = float(obs[0])
    dy_n = float(obs[1])
    return (dx_n * dx_n + dy_n * dy_n) ** 0.5 * max_distance_m


def select_action(actor, obs, noise_std=0.0):
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = actor(x).squeeze(0).cpu().numpy()

    if noise_std > 0.0:
        a_noised = np.random.normal(0.0, noise_std, size=a.shape) 
        #print("noise", a_noised, "a ", a, "a + noise ", a + a_noised)
        a = a + a_noised

    a = np.clip(a, -1.0, 1.0)
    return a.astype(np.float32)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def train_step(actor, critic, actor_tgt, critic_tgt, actor_opt, critic_opt, replay):
    s, a, r, s2, d = replay.sample(BATCH_SIZE)

    s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
    a = torch.tensor(a, dtype=torch.float32, device=DEVICE)
    r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    s2 = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
    d = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Critic update
    with torch.no_grad():
        a2 = actor_tgt(s2)
        y = r + GAMMA * (1.0 - d) * critic_tgt(s2, a2)

    q = critic(s, a)
    critic_loss = nn.SmoothL1Loss()(q, y)

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_opt.step()

    # Actor update
    pred_a = actor(s)
    actor_loss = -critic(s, pred_a).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_opt.step()

    # Soft target updates
    soft_update(actor_tgt, actor, TAU)
    soft_update(critic_tgt, critic, TAU)

    return float(actor_loss.item()), float(critic_loss.item())


def main():
    seed = DEFAULT_SEED
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"[WARN] invalid seed '{sys.argv[1]}', using {DEFAULT_SEED}")

    set_global_seed(seed)
    print(f"[INFO] seed = {seed}")

    client = JsonLineClient(HOST, PORT)
    client.connect()

    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)

    actor_tgt = Actor().to(DEVICE)
    critic_tgt = Critic().to(DEVICE)
    actor_tgt.load_state_dict(actor.state_dict())
    critic_tgt.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay = Replay()

    step = 0
    noise_std = NOISE_STD

    last_obs = None
    last_action = None

    best_reward_ma = float("-inf")
    recent_rewards = deque(maxlen=50)

    episode_idx = 0
    ep_reward = 0.0
    ep_len = 0
    actor_loss_sum = 0.0
    actor_loss_count = 0
    critic_loss_sum = 0.0
    critic_loss_count = 0
    final_distance = 0.0

    for msg in client.iter_messages():
        typ = msg.get("type")

        if typ == "reset":
            ep_reward = 0.0
            ep_len = 0
            actor_loss_sum = 0.0
            actor_loss_count = 0
            critic_loss_sum = 0.0
            critic_loss_count = 0

            obs = np.array(msg["obs"], dtype=np.float32)
            final_distance = estimate_distance_from_obs(obs, MAX_DISTANCE_METERS)

            action = select_action(actor, obs, noise_std=noise_std)
            client.send({
                "type": "action",
                "ax": float(action[0]),
                "ay": float(action[1]),
            })

            last_obs = obs
            last_action = action
            continue

        if typ == "step":
            obs2 = np.array(msg["obs"], dtype=np.float32)
            r = float(msg["reward"])
            done = 1 if msg["done"] else 0

            if last_obs is not None and last_action is not None:
                replay.push(last_obs, last_action, r, obs2, done)

            ep_reward += r
            ep_len += 1
            final_distance = estimate_distance_from_obs(obs2, MAX_DISTANCE_METERS)

            step += 1
            
            if len(replay) >= TRAIN_AFTER and step % TRAIN_EVERY == 0:
                a_loss, c_loss = train_step(
                    actor, critic, actor_tgt, critic_tgt,
                    actor_opt, critic_opt, replay
                )
                actor_loss_sum += a_loss
                actor_loss_count += 1
                critic_loss_sum += c_loss
                critic_loss_count += 1

            if step % 5000 == 0:
                torch.save(actor.state_dict(), "ddpg_actor_latest.pt")
                torch.save(critic.state_dict(), "ddpg_critic_latest.pt")
                print(f"[SAVE] latest checkpoint at step={step}")

            if done:
                avg_actor_loss = actor_loss_sum / max(actor_loss_count, 1)
                avg_critic_loss = critic_loss_sum / max(critic_loss_count, 1)

                append_training_log({
                    "episode": episode_idx,
                    "episode_reward": ep_reward,
                    "episode_length": ep_len,
                    "final_distance": final_distance,
                    "noise_std": noise_std,
                    "avg_actor_loss": avg_actor_loss,
                    "avg_critic_loss": avg_critic_loss,
                })

                recent_rewards.append(ep_reward)
                reward_ma = sum(recent_rewards) / len(recent_rewards)

                if len(recent_rewards) == recent_rewards.maxlen and reward_ma > best_reward_ma:
                    best_reward_ma = reward_ma
                    torch.save(actor.state_dict(), "ddpg_actor_best.pt")
                    torch.save(critic.state_dict(), "ddpg_critic_best.pt")
                    print(f"[SAVE] BEST model at episode={episode_idx} reward_MA50={reward_ma:.3f}")

                print(
                    f"episode={episode_idx} "
                    f"reward={ep_reward:.3f} "
                    f"len={ep_len} "
                    f"final_dist={final_distance:.3f} "
                    f"noise={noise_std:.3f} "
                    f"actor_loss={avg_actor_loss:.6f} "
                    f"critic_loss={avg_critic_loss:.6f}"
                )

                noise_std = max(NOISE_STD_MIN, noise_std * NOISE_DECAY)

                episode_idx += 1
                last_obs, last_action = None, None
                continue

            action2 = select_action(actor, obs2, noise_std=noise_std)
            client.send({
                "type": "action",
                "ax": float(action2[0]),
                "ay": float(action2[1]),
            })
            last_obs = obs2
            last_action = action2
            continue

        print("Unknown:", msg)


if __name__ == "__main__":
    main()
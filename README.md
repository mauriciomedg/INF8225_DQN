# INF8225_DQN

## Deep Q-Network Agent in Unreal Engine

This project demonstrates a reinforcement learning agent trained with a Deep Q-Network (DQN) to follow a moving player inside an Unreal Engine environment.

The agent receives observations from the simulation, selects movement actions using a trained policy, and attempts to stay close to the player over time.

## Inference Demo

The following video shows the trained DQN agent during inference.  
At this stage, the agent is no longer training; it is using the learned policy to follow the player in real time.

https://github.com/user-attachments/assets/77d40fff-d4e0-41f5-9ac9-2bc83778edce

## Project Overview

The goal of this project is to connect an Unreal Engine simulation with a Python reinforcement learning pipeline. The environment sends observations to the learning system, and the agent returns actions that control its movement.

Main components:

- Unreal Engine environment
- Python training script
- DQN-based action selection
- Replay buffer for experience storage
- Neural network approximation of the Q-function
- Real-time inference through socket communication

## Reinforcement Learning Setup

At each time step, the agent observes the relative state of the player and chooses an action. The objective is to maximize the long-term reward by reducing the distance to the player and avoiding failure conditions.

The DQN learns an action-value function:

```math
Q(s, a)






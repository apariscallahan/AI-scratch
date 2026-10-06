"""Reinforcement learning (``nb.rl``): simulated worlds, learning algorithms and replays.

Worlds (``World(kind, name=..., parts=[...])``): ``car`` (drive over terrain, pymunk physics),
``cartpole``, ``mountaincar``, ``pendulum``, ``lander``, ``maze`` and ``flappy``. World parts
(terrain, abilities, sensors, rewards, endings, …) are small setting blocks.

Algorithms (``train(model, world, algorithm=...)``): ``ppo`` (default), ``evolution``
(neuro-evolution), ``dqn`` and ``qtable`` (mazes). The model's layer blocks become the policy
network; what it learned is kept in ``model.agent``.
"""
from __future__ import annotations

from .agent import Agent
from .api import average_score, show_world, train, watch
from .parts import (Boost, Brake, CarBody, CustomReward, Drive, EndAfter, EndIfStuck, EndWhen, Fuel, Gap, Gravity,
                    Jump, Lean, MazeLayout, MazeSize, MazeTraps, MazeWalls, PipeSpeed, PoleLength, Reward, SameWorld,
                    Sense, Terrain, Wind)
from .play import play_yourself
from .world import WORLD_KINDS, World

__all__ = [
    "World", "train", "watch", "show_world", "average_score", "play_yourself", "Agent", "WORLD_KINDS",
    "Terrain", "SameWorld", "CarBody", "Drive", "Brake", "Lean", "Jump", "Boost", "Sense", "Reward", "CustomReward",
    "EndAfter", "EndIfStuck", "EndWhen", "Gravity", "MazeLayout", "MazeSize", "MazeWalls", "MazeTraps", "Wind",
    "Fuel", "Gap", "PipeSpeed", "PoleLength",
]

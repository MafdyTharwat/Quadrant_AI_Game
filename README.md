# Neural Reaction: AI-Powered Chain Reaction Game

**Neural Reaction** is a strategic grid-based board game where players compete to dominate the board through explosive chain reactions. This version features a sophisticated AI opponent that combines **Reinforcement Learning (PPO)** with **Heuristic Search Algorithms** to provide a challenging experience.

## Game Overview
The game is played on a 5x5 grid. Players place "atoms" on the cells. When a cell reaches its critical mass (4 atoms), it explodes, spreading atoms to adjacent cells and converting them to the player's color.

### Key Features:
* **Intelligent AI:** An opponent trained using Proximal Policy Optimization (PPO).
* **Heuristic Engine:** A custom evaluation function that analyzes board states for strategic moves.
* **Polished UI:** A smooth 60-FPS interface built with Pygame.
* **Hybrid Logic:** Combines neural network predictions with rule-based heuristics.

## Tech Stack
* **Language:** Python 3.12.7
* **AI Library:** Stable Baselines3 (PPO Algorithm)
* **Environment:** Gymnasium (Custom Reinforcement Learning Environment)
* **GUI:** Pygame
* **Math/Logic:** NumPy

## AI Strategy
* **Explosion Potential:** Prioritizes moves that trigger chain reactions.
* **Territory Control:** Calculates the gain of capturing opponent cells.
* **Threat Assessment:** Identifies and avoids cells under immediate threat by the player.

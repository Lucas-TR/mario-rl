# Mario RL – PPO on RAM

This repo trains a PPO agent to play **Super Mario Bros-1-1** using a RAM-based wrapper and then lets you watch the trained agent play.

## 1. Setup

From the project root:

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install stable-baselines3[extra] gym-super-mario-bros nes-py tensorboard

## 2. Training

To start training with the current config:

python train_mario_ram.py

Models are saved under:

- models/ppo_ram_notebook/<run_timestamp>/
- logs/ppo_ram_notebook/<run_timestamp>/

## 3. Visualizing training (TensorBoard)

In another terminal:

tensorboard --logdir logs/ppo_ram_notebook

Then open the URL that TensorBoard prints (e.g. http://localhost:6006).

## 4. Watching the agent play

Edit play_mario_ram.py if needed so that:

MODEL_CONFIGS["1"]["run_dir"]

points to the desired run folder.

Then run:

python play_mario_ram.py --episodes 3

Use --stochastic to enable stochastic actions:

python play_mario_ram.py --episodes 3 --stochastic

The environment will render in a window and print the total reward per episode.

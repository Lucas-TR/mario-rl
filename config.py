# config.py
from pathlib import Path
from datetime import datetime

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# ---------- EXPERIMENT NAME ----------
# Option A: set manually and change before each run:
# EXPERIMENT_NAME = "run_lr3e4_10M_01"

# Option B: automatic name with timestamp (recommended):
EXPERIMENT_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
# ------------------------------------

# Environment
ENV_ID = "SuperMarioBros-1-1-v0"

# Cropping and stack/skip (same as in the notebook)
X0, X1 = 0, 16
Y0, Y1 = 0, 13
N_STACK = 4
N_SKIP = 4

# PPO / training
TOTAL_TIMESTEPS = int(20e6)   # 20M steps
LEARNING_RATE = 3e-4
N_ENVS = 1

# Base directories for models and logs
BASE_MODEL_DIR = ROOT_DIR / "models" / "ppo_ram_notebook"
BASE_LOG_DIR   = ROOT_DIR / "logs" / "ppo_ram_notebook"

# Directories for THIS experiment
MODEL_DIR = BASE_MODEL_DIR / EXPERIMENT_NAME
LOG_DIR   = BASE_LOG_DIR / EXPERIMENT_NAME

# Base model name
MODEL_NAME = "ppo_ram_notebook"

# Callback config
CHECK_FREQ = 100_000      # save every 100k steps
SUCCESS_WINDOW = 100      # window for flag success rate
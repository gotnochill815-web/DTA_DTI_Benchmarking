import os
import random
import numpy as np
import torch


# ==========================================================
# Reproducibility
# ==========================================================

SEED = 32


def seed_everything(seed=SEED):
    """
    Make experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 256

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-5

NUM_EPOCHS = 200

PATIENCE = 20

MIN_DELTA = 1e-4


# ==========================================================
# Dataset
# ==========================================================

DATA_ROOT = "/content/GraphDTA-Benchmark/data"


# ==========================================================
# Checkpoints
# ==========================================================

CHECKPOINT_DIR = "/content/GraphDTA-Benchmark/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
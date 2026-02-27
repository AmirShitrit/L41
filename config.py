import torch

DATA_DIR = "data/mushrooms"
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

NUM_EPOCHS_FROZEN = 2
LR_FROZEN = 1e-3

NUM_EPOCHS_FINETUNE = 2
LR_FINETUNE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

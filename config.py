TRAIN_IMG_DIR = "data/COD10K/Train/Image"
TRAIN_MASK_DIR = "data/COD10K/Train/GT_Object"

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 200
THRESHOLD = 0.3
# IMG_SIZE = 256
NUM_WORKERS = 4
# NUM_SAMPLES = 1000
IMG_SIZE = 448

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "outputs/models/pranet (2).pth"
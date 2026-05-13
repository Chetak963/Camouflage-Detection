import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import CODDataset
from src.model import get_model
import config
import os

import torch.nn.functional as F

def edge_loss(pred, target):
    pred = torch.sigmoid(pred)

    laplace = torch.tensor(
        [[0,1,0],
         [1,-4,1],
         [0,1,0]], dtype=torch.float32
    ).view(1,1,3,3).to(pred.device)

    edge_pred = F.conv2d(pred, laplace, padding=1)
    edge_gt = F.conv2d(target, laplace, padding=1)

    return F.l1_loss(edge_pred, edge_gt)

def tversky_loss(pred, target, alpha=0.7, beta=0.3):
    pred = torch.sigmoid(pred)
    smooth = 1.

    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()

    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

# Combined Loss
def loss_fn(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    tv = tversky_loss(pred, target)
    edge = edge_loss(pred, target)

    return bce + 2 * tv + 0.7 * edge

def train():
    dataset = CODDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = get_model().to(config.DEVICE)

    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS): # One epoch = full dataset pass
        model.train() # Enables:
                        # Dropout
                        # BatchNorm training
        total_loss = 0

        # Batch loop
        for images, masks in loader: # DataLoader calls: __getitem__() → returns (image, mask)
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE).float()

            outputs = model(images)

            out, out2, out3, out4, out5 = outputs

            # 🔥 resize all to mask size
            out2 = F.interpolate(out2, size=masks.shape[2:], mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=masks.shape[2:], mode='bilinear', align_corners=False)
            out4 = F.interpolate(out4, size=masks.shape[2:], mode='bilinear', align_corners=False)
            out5 = F.interpolate(out5, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = (
                1.0 * loss_fn(out, masks) +
                1.0 * loss_fn(out2, masks) +
                0.8 * loss_fn(out3, masks) +
                0.6 * loss_fn(out4, masks) +
                0.4 * loss_fn(out5, masks)
                )
            # loss = criterion(outputs, masks)

            optimizer.zero_grad() # Backpropagation : Clear old gradients
            loss.backward() # Compute gradients (how wrong model is)
            optimizer.step() # Update weights

            total_loss += loss.item() # .item() → convert tensor → number

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Model saved!")

if __name__ == "__main__":
    train()
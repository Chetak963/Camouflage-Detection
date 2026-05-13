import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from src.model import get_model
import config


# Metrics
def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)

    return iou, dice


def evaluate():
    device = config.DEVICE

    # Load model
    model = get_model().to(device)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device),
        strict=False
    )
    model.eval()

    image_dir = config.TEST_IMG_DIR
    mask_dir = config.TEST_MASK_DIR

    image_list = sorted(os.listdir(image_dir))

    total_iou = 0
    total_dice = 0
    count = 0

    for img_name in tqdm(image_list):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            continue

        # 🔹 Load image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        image = image.astype("float32") / 255.0

        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # 🔹 Predict
        with torch.no_grad():
            output = model(tensor)

            if isinstance(output, tuple):
                pred_mask = output[0]
            else:
                pred_mask = output

            pred = torch.sigmoid(pred_mask)

        pred = pred.squeeze().cpu().numpy()

        # Resize prediction
        pred = cv2.resize(pred, (config.IMG_SIZE, config.IMG_SIZE))

        # Threshold
        pred = (pred > 0.2).astype(np.uint8)

        # Load GT mask
        gt = cv2.imread(mask_path, 0)
        gt = cv2.resize(gt, (config.IMG_SIZE, config.IMG_SIZE))
        gt = (gt / 255).astype(np.uint8)

        # Metrics
        iou, dice = compute_metrics(pred, gt)

        total_iou += iou
        total_dice += dice
        count += 1

    # Final results
    print("\n Evaluation Results")
    print(f"Images evaluated: {count}")
    print(f"Mean IoU:  {total_iou / count:.4f}")
    print(f"Mean Dice: {total_dice / count:.4f}")


if __name__ == "__main__":
    evaluate()
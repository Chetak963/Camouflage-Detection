import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import get_model
import config


def predict(image_path):
    device = config.DEVICE

    # Load model
    model = get_model().to(device)
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Read image
    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Image not found: {image_path}")

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Preprocess
    image = cv2.resize(orig, (config.IMG_SIZE, config.IMG_SIZE))
    image = image / 255.0

    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Predict
    with torch.no_grad():
        pred = model(tensor)
        pred = pred[0]   # take final output only
        pred = torch.sigmoid(pred)

        # Boost weak regions (important)
        pred = pred ** 0.3

    # Convert to numpy
    mask = pred.squeeze().cpu().numpy()

    # Smooth probabilities
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Threshold
    mask = (mask > mask.mean()).astype("uint8")
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

    # Morphology (fill + clean)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype("uint8")

    # Resize BACK to original size (IMPORTANT: do last)
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = orig_rgb.copy()
    overlay[mask == 1] = [0, 255, 0]
    print("mask", mask.shape)
    print("overlay", overlay.shape)

    # Show results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()
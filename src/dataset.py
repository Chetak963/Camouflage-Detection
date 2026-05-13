import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import config

class CODDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        all_images = sorted(os.listdir(image_dir))[:1000]

        self.images = []
        for img in all_images:
            mask_path = os.path.join(mask_dir, img.replace(".jpg", ".png"))
            if os.path.exists(mask_path):
                self.images.append(img)

        # define once
        self.transform = A.Compose([
            A.Resize(
                config.IMG_SIZE,
                config.IMG_SIZE,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RandomShadow(p=0.3),
            A.CLAHE(p=0.3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name) # images/cat.jpg
        # convert -> cat.jpg → cat.png and final -> masks/cat.png
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png")) 

        image = cv2.imread(img_path) # load images as -> (H, W, 3)  → BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR → RGB
        # OpenCV uses BGR, but PyTorch expects RGB
        mask = cv2.imread(mask_path, 0) # 0 = grayscale mode
        # o/p -> (H,W) , Pixel values: 0 → background,255 → object

  
        augmented = self.transform(image=image, mask=mask) # Using Albumentations
        image = augmented['image']
        mask = augmented['mask']

        image = image / 255.0 # Normalize image -> convert 0-255 -> 0-1

        # clean binary mask
        mask = (mask > 127).astype("float32") # Meaning -> pixel > 127 then 1
                                                        #          else 0

        image = torch.tensor(image).permute(2, 0, 1).float() # Convert image to tensor
        # (H, W, C) → (C, H, W)
        mask = torch.tensor(mask).unsqueeze(0).float() # Convert mask to tensor
        # unsqueeze(0) -> adds channel dimenstion
        # (H, W) → (1, H, W)
        return image, mask
        # Final output :
        # image → (3, H, W)
        # mask  → (1, H, W)
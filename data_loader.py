# data_loader.py

import os
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def rolling_ball_correction_rgb(image, radius=50):
    """
    Apply Rolling Ball background correction to an RGB image.
    """
    channels = cv2.split(image)
    corrected_channels = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    for channel in channels:
        background = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(channel, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        corrected_channels.append(corrected)
    corrected_image = cv2.merge(corrected_channels)
    return corrected_image

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, mask_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        # Resize images and masks to 256x256 using Albumentations
        self.resize = A.Resize(height=256, width=256)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        
        # Load image and apply rolling ball correction
        img = np.array(Image.open(img_path).convert("RGB"))
        img = rolling_ball_correction_rgb(img, radius=50)
        
        # Load mask and convert to binary
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask > 0] = 1
        
        # Resize
        resized = self.resize(image=img, mask=mask)
        img = resized["image"].astype(np.float32) / 255.0
        mask = resized["mask"]
        
        # Apply additional transforms if provided
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # Ensure mask has a channel dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return img, mask

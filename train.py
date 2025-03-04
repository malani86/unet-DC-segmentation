# train.py
# train.py
from PIL import Image
import cv2
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import SegmentationDataset
from model import UNet
from metrics import combined_loss, dice_coef, calculate_metrics, plot_binary_confusion_matrix_with_metrics

from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# Paths
image_dir = "/gpfs/home/malani/u_net/train_data/images/"
mask_dir = "/gpfs/home/malani/u_net/train_data/masks/"

# List and filter image files
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
])
mask_files = sorted([
    f for f in os.listdir(mask_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
])

# Ensure equal number of images and masks
assert len(image_files) == len(mask_files), "Mismatch between the number of images and masks!"

# Pair images and masks
image_mask_pairs = list(zip(image_files, mask_files))

# First, split into train+validation (80%) and test (20%)
train_val_pairs, test_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)

# From the train+val subset, allocate 25% to validation, leaving 75% for training.
# 25% of the 80% subset is 20% of the original dataset.
train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.25, random_state=42)

# Unzip pairs
train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
val_images, val_masks = zip(*val_pairs) if val_pairs else ([], [])
test_images, test_masks = zip(*test_pairs) if test_pairs else ([], [])

# Convert to lists
train_images, train_masks = list(train_images), list(train_masks)
val_images, val_masks = list(val_images), list(val_masks)
test_images, test_masks = list(test_images), list(test_masks)

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Testing set: {len(test_images)} images")

# Augmentations
train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    ToTensorV2()
])
val_transform = ToTensorV2()  # Typically just convert to tensor for val
test_transform = ToTensorV2() # Same for test

# Create datasets
train_dataset = SegmentationDataset(image_dir, mask_dir, train_images, train_masks, transform=train_augmentations, return_filename=True, return_orig_size=True)
val_dataset   = SegmentationDataset(image_dir, mask_dir, val_images,   val_masks,   transform=val_transform, return_filename=True ,return_orig_size=True)
test_dataset  = SegmentationDataset(image_dir, mask_dir, test_images,  test_masks,  transform=test_transform, return_filename=True ,return_orig_size=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------------
# Model Initialization
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)
model.to(device)
#print(model)

# -------------------------------
# Training Loop
# -------------------------------

num_epochs = 50
criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_dice = 0.0
patience = 10
patience_counter = 0

# Metric lists
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_dices = []
val_dices = []

for epoch in range(num_epochs):
    # -------- TRAINING --------
    model.train()
    train_loss = 0.0
    train_dice_sum = 0.0
    train_correct = 0
    train_total = 0
    
    for images, masks  ,orig_size, filenames in train_loader:
        images, masks = images.float().to(device), masks.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        predicted_masks = (outputs > 0.3).float().squeeze(1)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if predicted_masks.dim() == 3:
            predicted_masks = predicted_masks.unsqueeze(1)
        train_dice_sum += dice_coef(masks, predicted_masks).item()
        
        y_true_flat = masks.view(-1).cpu().numpy()
        y_pred_flat = predicted_masks.view(-1).cpu().numpy()
        train_correct += (y_true_flat == y_pred_flat).sum()
        train_total += len(y_true_flat)
        
    train_loss /= len(train_loader)
    train_dice_avg = train_dice_sum / len(train_loader)
    train_accuracy = train_correct / train_total
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_dices.append(train_dice_avg)

    # -------- VALIDATION --------
    model.eval()
    output_val_dir = "predicted_val_masks"
    os.makedirs(output_val_dir, exist_ok=True)

    val_loss = 0.0
    val_dice_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for i, batch in enumerate(val_loader):
            if len(batch) == 4:  # (img, mask,orig_size, filename)
                images, masks,orig_size, filenames = batch
            else:
                images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predicted_masks = (outputs > 0.3).float()
            loss = criterion(outputs, masks)
            val_loss += loss.item()
        # Binarize predictions at 0.5 threshold
            predicted_masks = (outputs > 0.3).float()
        # Dice metric
            val_dice_sum += dice_coef(masks, predicted_masks).item()

        # Pixel-wise accuracy
            y_true_flat = masks.view(-1).cpu().numpy()
            y_pred_flat = predicted_masks.view(-1).cpu().numpy()
            val_correct += (y_true_flat == y_pred_flat).sum()
            val_total += len(y_true_flat)
        
        preds = (outputs > 0.5).float().cpu().numpy()  # Define the `preds` variable here

        for j in range(images.size(0)):
            pred_mask_256 = preds[j].squeeze()
            if isinstance(orig_size, list) and all(isinstance(t, torch.Tensor) for t in orig_size):
                orig_size = list(zip(orig_size[0].cpu().numpy(), orig_size[1].cpu().numpy()))
           
            sample_orig_h = int(orig_size[j][0])
            sample_orig_w = int(orig_size[j][1])

            # e.g. using OpenCV (nearest-neighbor is common for masks):
            pred_mask_orig = cv2.resize(pred_mask_256, (sample_orig_w, sample_orig_h), interpolation=cv2.INTER_NEAREST)
            pred_mask_orig = (pred_mask_orig * 255).astype(np.uint8)

            # Save with the same filename as your target mask if you want
            base_name, _ = os.path.splitext(filenames[j])
            out_path = os.path.join("predicted_val_masks", f"{base_name}_pred.png")
            Image.fromarray(pred_mask_orig).save(out_path)
            print(f"Saved predicted mask: {out_path}")

    # Compute averages after the loop
    val_loss /= len(val_loader)
    val_dice_avg = val_dice_sum / len(val_loader)
    val_accuracy = val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_dices.append(val_dice_avg)

    print(f"Epoch {epoch+1}/{num_epochs} | "
      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
      f"Train Dice: {train_dice_avg:.4f}, Val Dice: {val_dice_avg:.4f}")
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    print("-------------------------------------------------------")

    # Early stopping check
    if val_dice_avg > best_dice:
        best_dice = val_dice_avg
        patience_counter = 0
        torch.save(model.state_dict(), "best_unet_model2.pth")
        print("Model saved!")
    else:
        patience_counter += 1
    if patience_counter >= patience:
            print("Early stopping!")
            break



""""  
    with torch.no_grad():
        for images, masks  in  val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            predicted_masks = (outputs > 0.5).float()
            val_dice_sum += dice_coef(masks, predicted_masks).item()
            
            y_true_flat = masks.view(-1).cpu().numpy()
            y_pred_flat = predicted_masks.view(-1).cpu().numpy()
            val_correct += (y_true_flat == y_pred_flat).sum()
            val_total += len(y_true_flat)

    val_loss /= len(val_loader)
    val_dice_avg = val_dice_sum / len(val_loader)
    val_accuracy = val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_dices.append(val_dice_avg)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Dice: {train_dice_avg:.4f}, Val Dice: {val_dice_avg:.4f}")
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    print("-------------------------------------------------------")
    
    # Early Stopping
    if val_dice_avg > best_dice:
        best_dice = val_dice_avg
        patience_counter = 0
        torch.save(model.state_dict(), "best_unet_model2.pth")
        print("Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# -------------------------------
# Final Testing + Visualization
# -------------------------------

# Load best model
model.load_state_dict(torch.load("best_unet_model2.pth"))
model.eval()

# Evaluate on the test set
test_loss = 0.0
test_dice_sum = 0.0
test_correct = 0
test_total = 0
all_y_true = []
all_y_pred = []

with torch.no_grad():
    for images, masks, mask_filenames in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        predicted_masks = (outputs > 0.5).float().cpu().numpy()
        loss = criterion(outputs, masks)
        test_loss += loss.item()

        predicted_masks = (outputs > 0.5).float()
        test_dice_sum += dice_coef(masks, predicted_masks).item()

        y_true_flat = masks.view(-1).cpu().numpy()
        y_pred_flat = predicted_masks.view(-1).cpu().numpy()
        test_correct += (y_true_flat == y_pred_flat).sum()
        test_total += len(y_true_flat)

        all_y_true.append(y_true_flat)
        all_y_pred.append(y_pred_flat)

test_loss /= len(test_loader)
test_dice_avg = test_dice_sum / len(test_loader)
test_accuracy = test_correct / test_total

from sklearn.metrics import accuracy_score
overall_test_accuracy = accuracy_score(np.concatenate(all_y_true), np.concatenate(all_y_pred))

print("========== Test Results ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice: {test_dice_avg:.4f}")
print(f"Test Accuracy (pixel-wise): {test_accuracy:.4f}")

# Confusion Matrix on Test
precision, recall, f1, specificity, conf_matrix = calculate_metrics(
    torch.tensor(np.concatenate(all_y_true)),
    torch.tensor(np.concatenate(all_y_pred))
)
from metrics import plot_binary_confusion_matrix_with_metrics
plot_binary_confusion_matrix_with_metrics(conf_matrix, overall_test_accuracy)
"""
# Additional Plots: Training vs. Validation
epochs_range = range(1, len(train_losses) + 1)

# Plot Loss & Dice
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', color='red')
plt.plot(epochs_range, val_losses[len(train_losses)*-1:], label='Validation Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_dices, label='Training Dice', color='red')
plt.plot(epochs_range, val_dices[len(train_dices)*-1:], label='Validation Dice', color='green')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.title('Training and Validation Dice')
plt.tight_layout()
plt.savefig("loss_and_dice_plot2.png")
"""

# Quick Visualization of Some Test Predictions
output_dir = "predicted_masks_test"
os.makedirs(output_dir, exist_ok=True)

import matplotlib
from PIL import Image

output_dir = "predicted_masks_test"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, (images, masks, mask_filenames) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        predicted_masks = (outputs > 0.5).float().cpu().numpy()
        for i in range(images.size(0)):
            # Save predicted mask
            pred_mask = predicted_masks[i].squeeze()
            pred_mask_img = (pred_mask * 255).astype(np.uint8)
            original_filename = mask_filenames[i]
            base_name, ext = os.path.splitext(original_filename)
            save_path = os.path.join(output_dir, f"{base_name}_pred.png")
            im = Image.fromarray(pred_mask_img)
            im.save(save_path)
            print(f"Saved predicted mask to: {save_path}")


        # For demonstration, let's visualize up to 3 images from the batch
        for i in range(min(3, images.size(0))):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i].cpu().permute(1, 2, 0))
            plt.title("Original Image")
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].squeeze(), cmap="gray")
            plt.title("True Mask")
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[i].squeeze(), cmap="gray")
            plt.title("Predicted Mask")
            plt.savefig(f"prediction_visualization_test_batch{batch_idx}_img{i}.png")
            plt.close()
"""
if __name__ == "__main__":
    pass



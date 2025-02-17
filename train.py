# train.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import SegmentationDataset
from model import UNet
from metrics import combined_loss, dice_coef, calculate_metrics, plot_binary_confusion_matrix_with_metrics

from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set paths
image_dir = "/Bureau/LaBRI_stage/u_net/train_data/images/"
mask_dir = "/Bureau/LaBRI_stage/u_net/train_data/masks/"

# List files
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Split dataset (ensure images and masks are paired)
image_mask_pairs = list(zip(image_files, mask_files))
train_pairs, test_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)
train_images, train_masks = zip(*train_pairs)
test_images, test_masks = zip(*test_pairs)
train_images, train_masks = list(train_images), list(train_masks)
test_images, test_masks = list(test_images), list(test_masks)

print(f"Training set: {len(train_images)} images")
print(f"Testing set: {len(test_images)} images")

# Define augmentations
train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    ToTensorV2()
])
test_transform = ToTensorV2()

# Create datasets and dataloaders
train_dataset = SegmentationDataset(image_dir, mask_dir, train_images, train_masks, transform=train_augmentations)
test_dataset = SegmentationDataset(image_dir, mask_dir, test_images, test_masks, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train DataLoader batches: {len(train_loader)}")
print(f"Test DataLoader batches: {len(test_loader)}")

# -------------------------------
# Model Initialization
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)
model.to(device)
print(model)

# Training loop
num_epochs = 200
criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_dice = 0.0
patience = 20
patience_counter = 0

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_dices = []
val_dices = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_dice_sum = 0.0
    train_correct = 0
    train_total = 0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted_masks = (outputs > 0.5).float()
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

    # Validation
    model.eval()
    val_loss = 0.0
    val_dice_sum = 0.0
    val_correct = 0
    val_total = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            predicted_masks = (outputs > 0.5).float()
            dice_val = dice_coef(masks, predicted_masks)
            val_dice_sum += dice_val.item()
            y_true_flat = masks.view(-1).cpu().numpy()
            y_pred_flat = predicted_masks.view(-1).cpu().numpy()
            val_correct += (y_true_flat == y_pred_flat).sum()
            val_total += len(y_true_flat)
            all_y_true.append(y_true_flat)
            all_y_pred.append(y_pred_flat)
            
    val_loss /= len(test_loader)
    val_dice_avg = val_dice_sum / len(test_loader)
    val_accuracy = val_correct / val_total
    overall_accuracy = accuracy_score(np.concatenate(all_y_true), np.concatenate(all_y_pred))
    
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_dices.append(val_dice_avg)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice_avg:.4f}")
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    print("-------------------------------------------------------")
    
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

# Load the best model for evaluation
model.load_state_dict(torch.load("best_unet_model2.pth"))
model.eval()

# Plot confusion matrix using the last validation batch results
from metrics import plot_binary_confusion_matrix_with_metrics
plot_binary_confusion_matrix_with_metrics(conf_matrix, overall_accuracy)

# Additional plotting (metrics over epochs, loss/dice plots, etc.)
# (You can reuse your existing plotting code here)
########################################
# Evaluation and Visualization
########################################

model.load_state_dict(torch.load("best_unet_model2.pth"))
model.eval()

plot_binary_confusion_matrix_with_metrics(all_conf_matrix, overall_accuracy)

# Plot Evaluation Metrics Over Time
epochs = list(range(1, len(val_epoch_precisions) + 1))
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_epoch_precisions, label="Precision")
plt.plot(epochs, val_epoch_recalls, label="Recall")
plt.plot(epochs, val_epoch_f1s, label="F1-Score")
plt.plot(epochs, val_epoch_specificities, label="Specificity")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Evaluation Metrics Over Time")
plt.savefig("evaluation_metrics2.png")
plt.show()

# Plot Loss and Dice curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.subplot(1, 2, 2)
plt.plot(val_dices, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()
plt.title("Validation Dice")
plt.savefig("loss_and_dice_plot2.png")
plt.close()

# Plot Metrics Subplots: Loss, Accuracy, and Dice
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(train_losses, label='Training Loss', color='red')
axes[0].plot(val_losses, label='Validation Loss', color='green')
axes[0].set_title('(a) Weighted Cross-Entropy Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(train_accuracies, label='Training Accuracy', color='red')
axes[1].plot(val_accuracies, label='Validation Accuracy', color='green')
axes[1].set_title('(b) Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[2].plot(train_dices, label='Training Dice', color='red')
axes[2].plot(val_dices, label='Validation Dice', color='green')
axes[2].set_title('(c) Dice')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Dice')
axes[2].legend()
plt.tight_layout()
plt.savefig("metrics_subplot.png")

# Evaluation on the test set (visualizing predictions)
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        outputs = model(images)
        predicted_masks = (outputs > 0.5).float().cpu().numpy()
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
            plt.savefig(f"prediction_visualization2_{i}.png")
            plt.close()


if __name__ == "__main__":
    # Optionally, add further evaluation or test-set visualization here.
    pass

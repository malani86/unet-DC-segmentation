# metrics.py

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def dice_loss(pred, target, smooth=1e-7):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_pred = (y_pred > 0.5).float()  # Binarize predictions
    intersection = (y_true * y_pred).sum(dim=(2, 3))
    union = y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    y_true_flat = y_true.view(-1).cpu().numpy()
    y_pred_flat = y_pred.view(-1).cpu().numpy()
    precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=1)
    recall = recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=1)
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=1)
    conf_matrix = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return precision, recall, f1, specificity, conf_matrix

def plot_binary_confusion_matrix_with_metrics(cm, accuracy):
    """
    Plot a binary confusion matrix (2x2) with per-class metrics on the diagonal,
    and overall accuracy in the title.
    """
    tn, fp, fn, tp = cm.ravel()
    # For class 0 ("Negative")
    pr0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    sp0  = tp / (tp + fp) if (tp + fp) > 0 else 0
    # For class 1 ("Positive")
    pr1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp1  = tn / (tn + fn) if (tn + fn) > 0 else 0

    annot = np.empty((2, 2), dtype=object)
    annot[0, 0] = f"{tn}\nPr={pr0:.2f}\nRec={rec0:.2f}\nSp={sp0:.2f}"
    annot[0, 1] = f"{fp}"
    annot[1, 0] = f"{fn}"
    annot[1, 1] = f"{tp}\nPr={pr1:.2f}\nRec={rec1:.2f}\nSp={sp1:.2f}"
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"Overall Accuracy: {accuracy:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_.png")
    

import os
import cv2
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from skimage.measure import label, regionprops_table

from model_2 import UNetDC
from data_loader import rolling_ball_correction_rgb

from algorithms import (
    contour_scan,
    calculate_contours_centroid,
    get_targets,
    density_maps,
    binary_to_dots,
    labeling_custom,
    calculate_centroids_sizes_image
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

def load_model(ckpt):
    m = UNetDC(in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.to(DEVICE).eval()

def preprocess(path: Path):
    im = np.array(Image.open(path).convert("RGB"))
    oh, ow = im.shape[:2]
    im = rolling_ball_correction_rgb(im, 50)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1), (oh, ow)

def quantify(bin_mask, min_area, px_per_um):
    lbl = label(bin_mask, connectivity=1)
    for l in np.unique(lbl):
        if l and (lbl == l).sum() < min_area:
            lbl[lbl == l] = 0
    lbl = label(lbl, connectivity=1)
    if lbl.max() == 0:
        return pd.DataFrame()
    props = regionprops_table(lbl, properties=["label", "area", "equivalent_diameter", "centroid"])
    df = pd.DataFrame(props)
    if px_per_um is not None and not df.empty:
        df["area_sqmicron"] = df["area"] / (px_per_um ** 2)
        df["eq_diam_micron"] = df["equivalent_diameter"] / px_per_um
    return df

def save_heatmap(img, out_path, cmap, vmin, vmax):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def compute_and_save_heatmaps(mask, orig_img, out_dir, name, kernel_size, nb_layers, contour_thresh, contour_min_size):
    roi_mask = contour_scan(orig_img, contour_thresh)
    roi_mask = label(roi_mask)
    roi_mask = np.isin(roi_mask, np.where(np.bincount(roi_mask.flat)[1:] >= contour_min_size)[0]+1)

    plt.imsave(out_dir / f"{name}_mask_contour_debug.png", roi_mask, cmap="gray")

    mask = mask & roi_mask
    dots = binary_to_dots(mask)
    labels = labeling_custom(mask, dots)
    centroid_size_img = calculate_centroids_sizes_image(dots, labels, roi_mask)

    centroid_y, centroid_x = calculate_contours_centroid(roi_mask)

    target_map, *_ = get_targets(
        mask, roi_mask, centroid_size_img, nb_layers, centroid_y, centroid_x
    )
    convoluted_map, *_ = density_maps(
        mask, roi_mask, centroid_size_img, kernel_size
    )

    save_heatmap(target_map, out_dir / f"{name}_target_density_heatmap_percentage.png", cmap="YlOrBr", vmin=0, vmax=35)
    save_heatmap(convoluted_map, out_dir / f"{name}_convoluted_density_heatmap_percentage.png", cmap="hot", vmin=0, vmax=15)

@torch.no_grad()
def run_batch(tensors, meta, model, mask_dir, overlay_dir, thresh, min_area, px_per_um, kernel_size, nb_layers, contour_thresh, contour_min_size):
    batch = torch.stack(tensors).to(DEVICE)
    logits = model(batch)
    for i, (fpath, (oh, ow)) in enumerate(meta):
        name = Path(fpath).stem
        mask_resized = (logits[i, 0].cpu().numpy() > thresh).astype(np.uint8)
        mask = cv2.resize(mask_resized, (ow, oh), cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_dir / f"{name}_pred.png"), mask * 255)

        orig_img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        compute_and_save_heatmaps(mask.astype(bool), orig_img, mask_dir, name, kernel_size, nb_layers, contour_thresh, contour_min_size)

        df = quantify(mask, min_area, px_per_um)
        df.insert(0, "filename", Path(fpath).name)
        df.to_csv(mask_dir.parent / f"{name}_droplets.csv", index=False)

        if overlay_dir:
            img = cv2.imread(fpath)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0,255,0), 2)
            cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--ckpt_path", default="best_UNetDC_focal_model.pth")
    parser.add_argument("--out_dir", default="quanti_results")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prob_thresh", type=float, default=0.3)
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--px_per_micron", type=float, required=True)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=55)
    parser.add_argument("--nb_layers", type=int, default=3)
    parser.add_argument("--contour_thresh", type=int, default=0)
    parser.add_argument("--contour_min_size", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "predicted_masks"
    overlay_dir = out_dir / "overlays" if args.save_overlays else None
    mask_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(exist_ok=True)

    model = load_model(args.ckpt_path)
    images = sorted(Path(args.img_dir).glob("*.*"))
    tensors, meta = [], []

    for img_path in tqdm(images):
        t, osize = preprocess(img_path)
        tensors.append(t)
        meta.append((str(img_path), osize))
        if len(tensors) == args.batch:
            run_batch(tensors, meta, model, mask_dir, overlay_dir, args.prob_thresh, args.min_area, args.px_per_micron, args.kernel_size, args.nb_layers, args.contour_thresh, args.contour_min_size)
            tensors, meta = [], []

    if tensors:
        run_batch(tensors, meta, model, mask_dir, overlay_dir, args.prob_thresh, args.min_area, args.px_per_micron, args.kernel_size, args.nb_layers, args.contour_thresh, args.contour_min_size)

    print("✅ All done! Outputs are in", out_dir)

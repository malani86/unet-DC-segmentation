# quantify_droplets_batch.py — inference + extended quantification report
# --------------------------------------------------------------------
# 1. Run UNetDC (batch inference) exactly as in train_DC_focal.py
# 2. Save per‑image predicted masks (+ optional overlays)
# 3. Save per‑droplet CSV per image **AND** a master CSV/Excel sheet
# 4. Plot a global size‑distribution histogram and compute summary stats
# 5. Compute spatial and radial density maps (BlobInspector style)
# --------------------------------------------------------------------

import os
import cv2
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from model_2 import UNetDC
from data_loader import rolling_ball_correction_rgb

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
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

def generate_roi_mask(img, blur_kernel=15):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, mask_contour = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    mask_contour = cv2.morphologyEx(mask_contour, cv2.MORPH_CLOSE, kernel)
    mask_contour = cv2.morphologyEx(mask_contour, cv2.MORPH_OPEN, kernel)
    return (mask_contour > 0).astype(np.uint8)

def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img



def get_targets(mask_thresh, mask_contour, nb_layers, centroid_y, centroid_x):
    """
    For each concentric ring, count the number of droplets whose centroid falls in that ring.
    Paint that count value onto all pixels in the ring for visualization.
    """
    lbl = label(mask_thresh, connectivity=1)
    props = regionprops_table(lbl, properties=["centroid"])
    cy_all, cx_all = props.get("centroid-0", []), props.get("centroid-1", [])

    coords = np.where(mask_contour)
    if len(coords[0]) == 0 or len(cx_all) == 0:
        return np.zeros_like(mask_thresh, dtype=np.float32)

    # Distance of all ROI pixels to centroid for ring definition
    distances = np.sqrt((coords[1] - centroid_x) ** 2 + (coords[0] - centroid_y) ** 2)
    max_distance = np.max(distances)
    ring_bounds = np.linspace(0, max_distance, nb_layers + 1)

    # Distance of each droplet centroid to center
    dists_centroids = np.sqrt((np.array(cx_all) - centroid_x) ** 2 + (np.array(cy_all) - centroid_y) ** 2)

    image_ring = np.zeros_like(mask_thresh, dtype=np.float32)
    for i in range(nb_layers):
        # Which droplets have centroids in this ring?
        in_ring = (ring_bounds[i] < dists_centroids) & (dists_centroids <= ring_bounds[i + 1])
        # Which ROI pixels belong to this ring?
        ring_mask = (ring_bounds[i] < distances) & (distances <= ring_bounds[i + 1])
        if np.any(ring_mask):
            count = np.sum(in_ring)
            image_ring[coords[0][ring_mask], coords[1][ring_mask]] = count
    return image_ring

def density_maps(mask_thresh, mask_contour, kernel_size=21):
    density_map = gaussian_filter(mask_thresh.astype(np.float32), sigma=kernel_size/6)
    density_map = density_map / (gaussian_filter(mask_contour.astype(np.float32), sigma=kernel_size/6) + 1e-5)
    density_map *= 100
    return density_map


@torch.no_grad()
def run_batch(tensors, meta, model, mask_dir, overlay_dir,
              thresh, min_area, px_per_um, per_image_rows, all_props):
    batch = torch.stack(tensors).to(DEVICE)
    logits = model(batch)
    for i in range(len(tensors)):
        fpath, (oh, ow) = meta[i]
        name = Path(fpath).stem
        mask512 = (logits[i, 0].cpu().numpy() > thresh).astype(np.uint8)
        mask    = cv2.resize(mask512, (ow, oh), cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_dir / f"{name}_pred.png"), mask * 255)

        df = quantify(mask, min_area, px_per_um)
        df.insert(0, "filename", Path(fpath).name)
        df.to_csv(mask_dir.parent / f"{name}_droplets.csv", index=False)
        all_props.append(df)

        per_image_rows.append({
            "filename": Path(fpath).name,
            "droplet_count": len(df),
            "total_area_px": df["area"].sum() if not df.empty else 0,
        })

        if overlay_dir is not None:
            img = cv2.imread(str(fpath))
            if img is not None:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), img)
            

        orig_img = np.array(Image.open(fpath).convert("RGB"))
        roi_mask = generate_roi_mask(orig_img)
        M = cv2.moments(roi_mask)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else ow // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else oh // 2

        radial_density = get_targets(mask, roi_mask, 10, cy, cx)
        spatial_density = density_maps(mask, roi_mask)
        

        plt.imsave(mask_dir.parent / f"{name}_radial_density.png", normalize(radial_density), cmap='hot')
        plt.imsave(mask_dir.parent / f"{name}_spatial_density.png", normalize(spatial_density), cmap='hot')

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

if __name__ == "__main__":
    p = argparse.ArgumentParser("Segment lipid droplets and build a report")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--ckpt_path", default="best_UNetDC_focal_model.pth")
    p.add_argument("--out_dir", default="quantify_results")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prob_thresh", type=float, default=0.3)
    p.add_argument("--min_area", type=int, default=1)
    p.add_argument("--px_per_micron", type=float)
    p.add_argument("--save_overlays", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "predicted_masks"
    overlay_dir = out_dir / "overlays" if args.save_overlays else None
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    if overlay_dir: overlay_dir.mkdir(exist_ok=True)

    model = load_model(args.ckpt_path)
    tensors, meta = [], []
    per_image_rows, all_props = [], []
    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])

    for img in tqdm(images, desc="Inference"):
        t, osize = preprocess(img)
        tensors.append(t)
        meta.append((str(img), osize))
        if len(tensors) == args.batch:
            run_batch(tensors, meta, model, mask_dir, overlay_dir,
                      args.prob_thresh, args.min_area, args.px_per_micron,
                      per_image_rows, all_props)
            tensors, meta = [], []

    if tensors:
        run_batch(tensors, meta, model, mask_dir, overlay_dir,
                  args.prob_thresh, args.min_area, args.px_per_micron,
                  per_image_rows, all_props)

    summary_df = pd.DataFrame(per_image_rows)
    summary_df.to_csv(out_dir / "summary_per_image.csv", index=False)

    if all_props:
        combined = pd.concat(all_props, ignore_index=True)
        combined.to_csv(out_dir / "all_droplets.csv", index=False)
        try:
            import xlsxwriter
            with pd.ExcelWriter(out_dir / "all_droplets.xlsx", engine="xlsxwriter") as xw:
                combined.to_excel(xw, index=False, sheet_name="droplets")
                summary_df.to_excel(xw, index=False, sheet_name="per_image")
        except (ImportError, AttributeError):
            combined.to_csv(out_dir / "all_droplets_noexcel.csv", index=False)
            print("⚠️  Skipped Excel file; install 'xlsxwriter<3.1.0' or use Python ≥3.7 if you need .xlsx output.")

        size_col = "eq_diam_micron" if "eq_diam_micron" in combined.columns else "equivalent_diameter"
        stats = combined[size_col].describe()[["mean", "50%", "std"]].rename({"50%": "median"})
        stats.to_csv(out_dir / "droplet_size_stats.csv")

        plt.figure(figsize=(6,4))
        plt.hist(combined[size_col], bins=40)
        plt.xlabel("Diameter (µm)" if "micron" in size_col else "Diameter (pixels)")
        plt.ylabel("Count")
        plt.title("Droplet size distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "size_histogram.png", dpi=300)
        plt.close()

    print("\n✓ All done. Outputs are in →", out_dir)

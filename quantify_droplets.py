import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops_table
from torchvision import transforms
from PIL import Image

# ---------- YOUR OWN UTILITIES -----------------
from model_2 import UNetDC                       # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
from data_loader import rolling_ball_correction_rgb  # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
# ------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512                    # you trained with A.Resize(512,512)

def load_model(ckpt_path):
    model = UNetDC(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def preprocess(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]

    img = rolling_ball_correction_rgb(img, radius=50)   # same as training
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_tensor  = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0)
    return img_tensor, (orig_h, orig_w)

@torch.no_grad()
def predict_mask(model, img_tensor):
    out = model(img_tensor.to(DEVICE))
    mask = (out[0,0].cpu().numpy() > 0.3).astype(np.uint8)   # same 0.3 thresh :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    return mask

def resize_to_orig(mask_512, orig_size):
    return cv2.resize(mask_512, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)

def quantify(mask_orig):
    labeled = label(mask_orig, connectivity=1)
    if labeled.max() == 0:
        return pd.DataFrame()   # no droplets
    props = regionprops_table(
        labeled,
        properties=["label", "area", "equivalent_diameter", "centroid"]
    )
    return pd.DataFrame(props)

def main(img_dir, ckpt_path="best_UNetDC_focal_model.pth", out_dir="quant_results",
         px_per_micron=None):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt_path)

    summary_rows = []
    for img_name in tqdm(sorted(os.listdir(img_dir))):
        if img_name.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
            img_path = os.path.join(img_dir, img_name)
            img_tensor, orig_size = preprocess(img_path)
            mask_512 = predict_mask(model, img_tensor)
            mask_orig = resize_to_orig(mask_512, orig_size)

            # -------- per-droplet --------
            props_df = quantify(mask_orig)
            if px_per_micron is not None and not props_df.empty:
                props_df["area_sqmicron"] = props_df["area"] / (px_per_micron**2)
                props_df["eq_diam_micron"] = props_df["equivalent_diameter"] / px_per_micron
            props_df.insert(0, "filename", img_name)
            props_df.to_csv(os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}_droplets.csv"),
                            index=False)

            # -------- per-image summary --------
            total_area = props_df["area"].sum() if not props_df.empty else 0
            droplet_count = len(props_df)
            summary_rows.append({"filename": img_name,
                                 "droplet_count": droplet_count,
                                 "total_area_px": total_area})

            # optional: save the mask overlay for quick QC
            overlay = cv2.addWeighted(
                cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                0.7,
                cv2.applyColorMap(mask_orig*255, cv2.COLORMAP_JET),
                0.3, 0)
            cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # -------- master summary --------
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary_per_image.csv"),
                                      index=False)
    print(f"Finished. Detailed CSVs and overlays in: {out_dir}")

if __name__ == "__main__":
    import argparse, textwrap
    parser = argparse.ArgumentParser(
        description="Segment lipid droplets and quantify them.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
                python quantify_droplets.py --img_dir path/to/new_images --px_per_micron 3.45
            """))
    parser.add_argument("--img_dir", required=True, help="Folder with .png/.jpg/.tif images")
    parser.add_argument("--ckpt_path", default="best_UNetDC_focal_model.pth")
    parser.add_argument("--out_dir", default="quant_results")
    parser.add_argument("--px_per_micron", type=float,
                        help="Pixel size (pixels per micron) for physical-unit output")
    args = parser.parse_args()
    main(**vars(args))


# Lipid Droplet Segmentation and Quantification using U-Net-DC

This project performs automated segmentation and quantification of lipid droplets in microscopy images using a customized U-Net-based model (U-Net-DC). It supports batch processing, statistical analysis, and optional overlay visualizations.

## 🧠 Model

The U-Net-DC architecture includes dilated convolutions to enhance the model's ability to detect small, scattered droplets with greater spatial awareness.

## 📂 Sample Project Structure

```plaintext
LipidDropletSegmentation/
│
├── models/                   # Model definition files
├── utils/                    # Utility scripts (data_loader.py, metrics_DC.py)
├── outputs/                  # Directory created to save results
│   ├── predicted_masks/      # Binary mask images
│   ├── overlays/             # segmentation overlay images
│   ├── summary_per_image.csv
│   ├── all_droplets.csv
│   ├── all_droplets.xlsx
│   ├── size_histogram.png
│   └── droplet_size_stats.csv
├── quantify_droplets_batch.py
├── train.py
├── train_DC_focal.py
├── install_packages.py
└── README.md
```

## 🛠️ Installation

Install dependencies:

 run:

```bash
python install_packages.py
```

## 🚀 Running the Pipeline

```bash
python quantify_droplets_batch.py \
  --img_dir input_images/ \
  --ckpt_path checkpoints/best_UNetDC_focal_model.pth \
  --out_dir outputs/ \
  --batch 8 \
  --prob_thresh 0.3 \
  --min_area 1 \
  --px_per_micron 5.0 \
  --save_overlays
```

## 📄 Output Description

- `summary_per_image.csv`: Count and area stats per image
- `*_droplets.csv`: Detailed droplet data per image
- `all_droplets.csv`: Aggregated droplet measurements
- `size_histogram.png`: Droplet size distribution plot
- `droplet_size_stats.csv`: Summary stats (mean, median, std)

## 🖼 Example

 example images showing before/after segmentation.


## 📬 Contact

For questions or collaboration, open an issue or contact the repository maintainer.

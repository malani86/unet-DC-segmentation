import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
import os


def browse_dir(var):
    path = filedialog.askdirectory()
    if path:
        var.set(path)


def browse_file(var):
    path = filedialog.askopenfilename()
    if path:
        var.set(path)


def run_process(img_dir_var, ckpt_var, out_dir_var, batch_var, prob_var,
                min_area_var, px_var, save_var, progress):
    if not img_dir_var.get() or not ckpt_var.get() or not out_dir_var.get():
        messagebox.showerror("Error", "Please fill in all required fields")
        return

    args = [
        "python",
        os.path.join(os.path.dirname(__file__), "quantify_droplets_batch.py"),
        "--img_dir", img_dir_var.get(),
        "--ckpt_path", ckpt_var.get(),
        "--out_dir", out_dir_var.get(),
        "--batch", str(batch_var.get()),
        "--prob_thresh", str(prob_var.get()),
        "--min_area", str(min_area_var.get()),
    ]
    if px_var.get():
        args += ["--px_per_micron", str(px_var.get())]
    if save_var.get():
        args.append("--save_overlays")

    def worker():
        try:
            subprocess.run(args, check=True)
            message = "Processing complete"
            messagebox.showinfo("Done", message)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Error", str(exc))
        finally:
            progress.stop()

    progress.start()
    threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    root.title("Droplet Quantification")

    img_dir_var = tk.StringVar()
    ckpt_var = tk.StringVar(value="best_UNetDC_focal_model.pth")
    out_dir_var = tk.StringVar(value="quant_results")
    batch_var = tk.IntVar(value=8)
    prob_var = tk.DoubleVar(value=0.3)
    min_area_var = tk.IntVar(value=1)
    px_var = tk.DoubleVar()
    save_var = tk.BooleanVar()

    fields = [
        ("Image directory", img_dir_var, browse_dir),
        ("Checkpoint path", ckpt_var, browse_file),
        ("Output directory", out_dir_var, browse_dir),
    ]

    for i, (label, var, browse) in enumerate(fields):
        tk.Label(root, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(root, textvariable=var, width=40)
        entry.grid(row=i, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=lambda v=var, b=browse: b(v)).grid(row=i, column=2, padx=5, pady=5)

    # Numeric fields
    numeric = [
        ("Batch size", batch_var),
        ("Probability threshold", prob_var),
        ("Minimum area", min_area_var),
        ("Pixels per micron", px_var),
    ]
    offset = len(fields)
    for i, (label, var) in enumerate(numeric):
        tk.Label(root, text=label).grid(row=offset+i, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(root, textvariable=var).grid(row=offset+i, column=1, padx=5, pady=5, sticky="w")

    tk.Checkbutton(root, text="Save overlays", variable=save_var).grid(row=offset+len(numeric), column=1, sticky="w", padx=5, pady=5)

    progress = ttk.Progressbar(root, mode="indeterminate")
    progress.grid(row=offset+len(numeric)+1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    run_btn = tk.Button(
        root, text="Run",
        command=lambda: run_process(img_dir_var, ckpt_var, out_dir_var,
                                    batch_var, prob_var, min_area_var, px_var,
                                    save_var, progress)
    )
    run_btn.grid(row=offset+len(numeric)+2, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()

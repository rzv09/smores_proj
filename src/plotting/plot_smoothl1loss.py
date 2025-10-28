import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os

def plot_smooth_l1_loss(csv_filepath: str, save_dir: str):
    
    # Load data
    df = pd.read_csv(csv_filepath)

    # (Optional) parse train_loss so ast doesn't complain if you reuse code later
    if "train_loss" in df.columns:
        df["train_loss"] = df["train_loss"].apply(ast.literal_eval)

    # Aggregate: mean SmoothL1Loss per model
    # Note: the column name has parentheses in it, keep it exactly
    loss_col = "SmoothL1Loss()"
    agg = df.groupby("model")[[loss_col]].mean().reset_index()

    # Bar positions
    x = np.arange(len(agg["model"]))
    width = 0.6

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, agg[loss_col], width)

    # Labels / styling
    ax.set_xlabel("Model")
    ax.set_ylabel("SmoothL1Loss (mean across datasets)")
    ax.set_title("Average SmoothL1Loss per Model")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model"])
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save and show
    png_filename = os.path.join(save_dir, f'avg_smoothL1_loss.png')
    plt.savefig(png_filename)
    print(f'Plot saved as {png_filename}')

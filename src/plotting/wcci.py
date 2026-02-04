import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

# -----------------------------
# Config / helpers
# -----------------------------
def nice_label(s: str) -> str:
    """fl1 -> FL1, sb3 -> SB3; otherwise return as-is."""
    s = str(s)
    m = re.fullmatch(r"(fl|sb)(\d+)", s)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return s


def parse_step_and_train_from_model(model_str: str) -> tuple[int, str]:
    """
    Supports:
      - TL: 'step3_train_fl2'  -> (3, 'FL2')
      - FL: 'step3_clients_fl1-fl2-fl3' -> (3, 'FL3')  # last added client
    """
    model_str = str(model_str)

    m_step = re.search(r"step(\d+)", model_str)
    if not m_step:
        raise ValueError(f"Could not parse step from model string: {model_str}")
    step = int(m_step.group(1))

    # TL format
    m_tl = re.search(r"train_([a-z]+\d+)", model_str)
    if m_tl:
        return step, nice_label(m_tl.group(1))

    # FL incremental format: stepK_clients_fl1-fl2-...-flK
    m_fl = re.search(r"clients_([a-z0-9\-]+)", model_str)
    if m_fl:
        client_list = m_fl.group(1).split("-")
        last_added = client_list[-1]
        return step, nice_label(last_added)

    raise ValueError(f"Unrecognized model string format: {model_str}")



@dataclass
class PlotPaths:
    heatmap_path: str
    prev_mean_path: str
    max_forgetting_bar_path: str
    summary_csv_path: str


# -----------------------------
# Core analysis
# -----------------------------
def make_matrices(
    df: pd.DataFrame,
    metric_col: str,
    mode: str,
    agg: str = "mean",
) -> Tuple[pd.DataFrame, List[int], List[str], List[str]]:
    """
    Returns:
      mat: rows=eval dataset/client, cols=step (1..T)
      steps: sorted steps
      x_labels: labels for x-axis (depends on mode)
      row_order: order of rows (we align with the training/addition order)
    """
    # parse step + trained_on from model
    parsed = df["model"].apply(parse_step_and_train_from_model)
    df = df.copy()
    df["step"] = parsed.apply(lambda x: x[0])
    df["trained_on"] = parsed.apply(lambda x: x[1])
    df["dataset_lbl"] = df["dataset"].apply(nice_label)

    # training/add order
    order_steps = (
        df.sort_values("step")
        .drop_duplicates("step")[["step", "trained_on"]]
        .sort_values("step")
    )
    steps = order_steps["step"].tolist()
    add_order = order_steps["trained_on"].tolist()  # order in which dataset/client is introduced

    # build x-axis labels
    if mode == "tl":
        # each stage is "last trained on dataset"
        x_labels = add_order
    elif mode == "fl_incremental":
        # each stage is "cumulative clients included"
        # seen = []
        # x_labels = []
        # for c in add_order:
        #     seen.append(c)
        #     x_labels.append("+".join(seen))
        x_labels = add_order
    else:
        raise ValueError("mode must be 'tl' or 'fl_incremental'")

    # aggregate duplicates (e.g., multiple seeds) -> pivot_table
    mat = df.pivot_table(
        index="dataset_lbl",
        columns="step",
        values=metric_col,
        aggfunc=agg,
    )

    # row order aligned with order of introduction
    row_order = add_order[:]
    mat = mat.reindex(row_order)

    return mat, steps, x_labels, row_order


def compute_max_forgetting(
    mat: pd.DataFrame,
    steps: List[int],
    intro_order: List[str],
) -> pd.DataFrame:
    """
    For each dataset/client introduced at step k:
      reference = metric at step k (right after introduced/trained)
      max_forgetting = max_{t>=k} (metric_t - metric_k)
    """
    records = []
    for i, d in enumerate(intro_order):
        learn_step = i + 1
        init_val = mat.loc[d, learn_step]
        later_steps = [s for s in steps if s >= learn_step]
        later_vals = mat.loc[d, later_steps].dropna()

        inc = later_vals - init_val
        records.append({
            "item": d,
            "learn_step": learn_step,
            "val_at_learn": float(init_val),
            "val_final": float(later_vals.iloc[-1]),
            "max_increase": float(inc.max()),
            "step_of_max_increase": int(inc.idxmax()),
            "final_change": float(later_vals.iloc[-1] - init_val),
        })

    return pd.DataFrame(records)

def plot_heatmap(
    mat,
    x_labels,
    row_labels,
    title,
    xlabel,
    ylabel,
    out_path,
    cmap: str = "coolwarm",          # nicer default than RdBu_r for readability
    annotate_fmt: str = "{:.4f}",
    font_scale: float = 1.0,
    clip_percentiles: tuple | None = (2, 98),  # set to None to disable clipping
    dynamic_text_color: bool = True,
    text_outline: bool = True,
):
    plt.rcParams.update({
        "font.size": 13 * font_scale,
        "axes.titlesize": 16 * font_scale,
        "axes.labelsize": 14 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 12 * font_scale,
    })

    heat = mat.values
    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        raise ValueError("Heatmap contains no finite values.")

    # Optional clipping helps avoid extreme dark/bright cells dominating the scale
    if clip_percentiles is None:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        lo, hi = clip_percentiles
        vmin, vmax = np.percentile(finite, lo), np.percentile(finite, hi)
        # guard against degenerate cases
        if vmin == vmax:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))

    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(12.0, 5.4))
    ax = plt.gca()
    im = ax.imshow(heat, aspect="auto", cmap=cm, norm=norm)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=25, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # annotate only finite cells
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            if not np.isfinite(val):
                continue

            txt_color = "black"
            if dynamic_text_color:
                r, g, b, _ = cm(norm(val))
                luminance = 0.299*r + 0.587*g + 0.114*b
                txt_color = "white" if luminance < 0.5 else "black"

            t = ax.text(
                j, i, annotate_fmt.format(val),
                ha="center", va="center",
                fontsize=9 * font_scale,
                color=txt_color
            )

            # Optional outline stroke makes labels readable everywhere
            if text_outline:
                outline_color = "black" if txt_color == "white" else "white"
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=outline_color)])

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=12 * font_scale)
    cbar.set_label(mat.columns.name if mat.columns.name else "", fontsize=12 * font_scale)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)



def plot_prev_mean_curve(
    mat: pd.DataFrame,
    steps: List[int],
    intro_order: List[str],
    x_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    mark_max: bool = True,
    font_scale: float = 1.0,
):
    plt.rcParams.update({
        "font.size": 13 * font_scale,
        "axes.titlesize": 16 * font_scale,
        "axes.labelsize": 14 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 12 * font_scale,
    })

    mean_prev = []
    for t in steps:
        prev_items = intro_order[:t-1]  # previously introduced
        vals = [mat.loc[item, t] for item in prev_items if np.isfinite(mat.loc[item, t])]
        mean_prev.append(float(np.mean(vals)) if vals else np.nan)

    fig = plt.figure(figsize=(10.5, 4.2))
    ax = plt.gca()
    ax.plot(np.arange(len(steps)), mean_prev, marker="o")

    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(x_labels, rotation=25, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if mark_max and np.any(np.isfinite(mean_prev)):
        idx = int(np.nanargmax(mean_prev))
        ax.axvline(idx, linestyle="--", linewidth=1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_max_forgetting_bar(
    forget_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    font_scale: float = 1.0,
):
    plt.rcParams.update({
        "font.size": 13 * font_scale,
        "axes.titlesize": 16 * font_scale,
        "axes.labelsize": 14 * font_scale,
        "xtick.labelsize": 12 * font_scale,
        "ytick.labelsize": 12 * font_scale,
    })

    fig = plt.figure(figsize=(8.6, 3.9))
    ax = plt.gca()
    ax.bar(forget_df["item"], forget_df["max_increase"], edgecolor="black", linewidth=0.6)
    ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# -----------------------------
# Public entrypoint
# -----------------------------
def analyze_and_plot(
    csv_path: str,
    mode: str = "tl",             # "tl" or "fl_incremental"
    metric: str = "MARE",         # "MARE" or "SmoothL1Loss()"
    out_dir: str = "./plots_out",
    agg: str = "mean",            # how to aggregate duplicates (e.g., seeds)
    font_scale: float = 1.0,
) -> PlotPaths:
    """
    Reads csv_path and generates:
      - heatmap (triangular matrix)
      - mean previous curve
      - max forgetting bar plot
      - summary csv of forgetting metrics

    Returns file paths to outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    mat, steps, x_labels, intro_order = make_matrices(df, metric_col=metric, mode=mode, agg=agg)

    # compute max-forgetting/degradation
    forget_df = compute_max_forgetting(mat, steps, intro_order)

    # output paths
    base = os.path.splitext(os.path.basename(csv_path))[0]
    heatmap_path = os.path.join(out_dir, f"{base}_{mode}_{metric}_heatmap.png")
    prev_mean_path = os.path.join(out_dir, f"{base}_{mode}_{metric}_prev_mean.png")
    bar_path = os.path.join(out_dir, f"{base}_{mode}_{metric}_max_forgetting_bar.png")
    summary_csv_path = os.path.join(out_dir, f"{base}_{mode}_{metric}_forgetting_summary.csv")

    # plot titles
    if mode == "tl":
        xlab = "Dataset last trained on"
        title_prefix = "Sequential TL"
    else:
        xlab = "Clients included in federation (cumulative)"
        title_prefix = "Incremental FL"

    plot_heatmap(
        mat=mat,
        x_labels=x_labels,
        row_labels=intro_order,
        title=f"{title_prefix}: {metric} across stages",
        xlabel=xlab,
        ylabel="Evaluation dataset/client",
        out_path=heatmap_path,
        cmap="RdBu_r",
        annotate_fmt="{:.4f}",
        font_scale=font_scale,
    )

    plot_prev_mean_curve(
        mat=mat,
        steps=steps,
        intro_order=intro_order,
        x_labels=x_labels,
        title=f"{title_prefix}: mean {metric} on previously learned clients/datasets",
        xlabel=xlab,
        ylabel=f"Mean {metric} on previous clients/datasets",
        out_path=prev_mean_path,
        mark_max=True,
        font_scale=font_scale,
    )

    plot_max_forgetting_bar(
        forget_df=forget_df,
        title=f"{title_prefix}: max degradation per client/dataset ({metric})",
        xlabel="Client/Dataset",
        ylabel=f"Max {metric} increase after introduction",
        out_path=bar_path,
        font_scale=font_scale,
    )

    # save summary
    forget_df.to_csv(summary_csv_path, index=False)

    return PlotPaths(
        heatmap_path=heatmap_path,
        prev_mean_path=prev_mean_path,
        max_forgetting_bar_path=bar_path,
        summary_csv_path=summary_csv_path,
    )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example:
    paths = analyze_and_plot(
        csv_path="/Users/ramanzatsarenko/smores_proj/out/cross_region_federated_forgetting_fedprox_1runs_2026-01-30_21-32-12/metrics.csv",
        mode="fl_incremental",
        metric="MARE",
        out_dir="plots_out",
        agg="mean",
        font_scale=1.1
    )
    print(paths)
    pass

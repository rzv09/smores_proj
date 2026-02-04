import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Inputs: summary CSVs produced by the script ---
tl_summary_csv = "/Users/ramanzatsarenko/smores_proj/plots_out/metrics_tl_MARE_forgetting_summary.csv"
fl_summary_csv = "/Users/ramanzatsarenko/smores_proj/plots_out/FL_metrics_fl_incremental_MARE_forgetting_summary.csv"

# --- Load ---
tl = pd.read_csv(tl_summary_csv)
fl = pd.read_csv(fl_summary_csv)

# The per-item name column is "item" in the script output
tl = tl.set_index("item")
fl = fl.set_index("item")

# Ensure consistent ordering on x-axis
order = ["FL1","FL2","FL3","FL4","SB1","SB2","SB3","SB4"]
tl = tl.reindex(order)
fl = fl.reindex(order)

x = np.arange(len(order))

# Values to plot (max forgetting / max degradation)
tl_vals = tl["max_increase"].values
fl_vals = fl["max_increase"].values

plt.figure(figsize=(10, 4))

# FL first (blue), TL on top (red) so TL stands out
plt.bar(x, fl_vals, width=0.72, color="tab:green", alpha=0.70, label="FL max forgetting", edgecolor="black", linewidth=0.6, zorder=2)
plt.bar(x, tl_vals, width=0.72, color="tab:blue",  alpha=0.55, label="TL max forgetting", edgecolor="black", linewidth=0.6, zorder=3)

plt.axhline(0, color="black", linewidth=1, zorder=1)
plt.xticks(x, order)
plt.yscale("log")
plt.ylabel("Max MARE increase (log scale)")
# plt.ylabel("Max MARE increase after introduction")
plt.title("Max forgetting per client/dataset (MARE): TL vs FL (overlaid)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_summary(path, label):
    s = pd.read_csv(path)
    s["experiment"] = label
    # convenience columns
    s["final_change"] = s["val_final"] - s["val_at_learn"]
    return s

# --- paths to the summary csv outputs from the script ---
tl_csv = "/Users/ramanzatsarenko/smores_proj/plots_out/metrics_tl_MARE_forgetting_summary.csv"
fl_csv = "/Users/ramanzatsarenko/smores_proj/plots_out/FL_metrics_fl_incremental_MARE_forgetting_summary.csv"

tl = load_summary(tl_csv, "TL (sequential)")
fl = load_summary(fl_csv, "FL (incremental)")

# Ensure same item ordering on the x-axis
order = ["FL1","FL2","FL3","FL4", "SB1","SB2","SB3","SB4"] 
# order = ["SB1","SB2","SB3","SB4", "FL1","FL2","FL3","FL4"]  # customize if needed
tl = tl.set_index("item").reindex(order).reset_index()
fl = fl.set_index("item").reindex(order).reset_index()

x = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10, 4))

# Curve 1: max forgetting
ax.plot(x, tl["max_increase"], marker="o", linewidth=2, label="TL max forgetting")
ax.plot(x, fl["max_increase"], marker="o", linewidth=2, label="FL max forgetting")

ax.axhline(0, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("Max increase after introduction")
ax.set_title("Max forgetting per dataset/client: TL vs FL")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from time import strftime, localtime

# --- settings ---

outcsv_dir = "CSVperOptode_2"

# add the slash and wildcard properly
pattern = os.path.join(outcsv_dir, "alldata_*.csv")
fnames = sorted(glob.glob(pattern))

print(fnames)  # check which files are found

if not fnames:
    raise FileNotFoundError(f"No files matched: {pattern}")

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

# colormap setup
n_lines = len(fnames)
cmap = mpl.cm.get_cmap("jet", n_lines)
norm = mpl.colors.Normalize(vmin=0, vmax=n_lines - 1)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # needed for the colorbar

labels = []
t0_for_label = None  # for x-label timestamp

for ii, fn in enumerate(fnames):
    df = pd.read_csv(fn)

    # basic column sanity check
    if "Epoch" not in df.columns or "DO" not in df.columns:
        print(f"Skipping {fn} (missing 'Epoch' or 'DO' columns)")
        continue
    if df.empty:
        print(f"Skipping {fn} (empty file)")
        continue

    if t0_for_label is None:
        # keep the very first file's start epoch for axis label
        t0_for_label = int(df["Epoch"].iloc[0])

    # plot this file
    ax.plot(
        # df["Epoch"] - df["Epoch"].iloc[0],
        pd.to_datetime(df["Epoch"] - df["Epoch"].iloc[0], unit='s'),
        df["DO"],
        ".",
        label=os.path.basename(fn).split(".csv")[0],  # label with filename (no extension)
        c=cmap(ii),
        markersize=3,
    )
    labels.append(os.path.basename(fn).split(".csv")[0])

# axis labels & colorbar
if t0_for_label is None:
    raise RuntimeError("No valid data rows found across files.")

ax.set_xlabel("Seconds since " + strftime("%Y-%m-%d %H:%M:%S", localtime(t0_for_label)))
ax.set_ylabel("DO Reading")

# Colorbar with integer ticks mapping to line order
cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(n_lines))
cbar.ax.set_yticklabels([str(i) for i in range(n_lines)])
cbar.set_label("File index (plot order)")

# Legend (can be large; feel free to remove or adjust)
ax.legend(loc="best", fontsize="small")

plt.tight_layout()
plt.show()

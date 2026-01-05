import os
import csv 
import pandas as pd

def write_csv(model_lbl: str, ds_lbl: str, seed: int, relative_error: float,  relative_error_baseline: float,
              criterion_type, criterion_val: float, criterion_val_baseline: float, loss_history: list, save_dir: str):
    csv_file = "metrics.csv"
    file_path = os.path.join(save_dir, csv_file)
    file_exists = os.path.isfile(file_path)
    # Append error value to the CSV file
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["model", "dataset", "seed", f"{criterion_type}", f"{criterion_type}_baseline", "MARE",
                              "MARE_baseline", "train_loss"])

        writer.writerow([model_lbl, ds_lbl, seed, criterion_val, criterion_val_baseline, relative_error, relative_error_baseline,
                          loss_history])
    return file_path


def write_stats_mare(csv_filepath:  str, save_dir: str):
    file_name = 'MARE_stats.csv'
    file_path = os.path.join(save_dir, file_name)
    df = pd.read_csv(csv_filepath)
    result = df.groupby(["model", "dataset"])["MARE"].agg(["mean", "std"]).reset_index()
    result.to_csv(file_path, index=False)

def write_stats_crit(csv_filepath: str, save_dir: str, criterion):
    file_name = 'CRIT_stats.csv'
    file_path = os.path.join(save_dir, file_name)
    df = pd.read_csv(csv_filepath)
    result = df.groupby(["model", "dataset"])[f"{criterion}"].agg(["mean", "std"]).reset_index()
    result.to_csv(file_path, index=False)

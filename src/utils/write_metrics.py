import os
import csv 
import pandas as pd

def write_csv(model_lbl: str, ds_lbl: str, seed: int, error: float, loss: list, save_dir: str):
    csv_file = "metrics.csv"
    file_path = os.path.join(save_dir, csv_file)
    file_exists = os.path.isfile(file_path)
    # Append error value to the CSV file
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["model", "dataset", "seed", "MARE", "train_loss"])

        writer.writerow([model_lbl, ds_lbl, seed, error, loss])
    return file_path


def write_stats(csv_filepath:  str, save_dir: str):
    file_name = 'stats.csv'
    file_path = os.path.join(save_dir, file_name)
    df = pd.read_csv(csv_filepath)
    result = df.groupby(["model", "dataset"])["MARE"].agg(["mean", "std"]).reset_index()
    result.to_csv(file_path, index=False)

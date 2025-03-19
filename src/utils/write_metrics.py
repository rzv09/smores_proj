import os
import csv 

def write_csv(prefix: str, ds_lbl: str, error: float, save_dir: str):
    csv_file = "metrics.csv"
    file_path = os.path.join(save_dir, csv_file)
    file_exists = os.path.isfile(file_path)
    # Append error value to the CSV file
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["prefix", "dataset", "MARE"])

        writer.writerow([prefix, ds_lbl, error])
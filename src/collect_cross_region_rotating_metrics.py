import pandas as pd

CSV_PATH = "/Users/ramanzatsarenko/smores_proj/out/cross_region_rotating_1runs_2026-01-28_14-45-30/metrics.csv"
COLUMNS = ["MARE", "MARE_baseline", "SmoothL1Loss()"]


def main():
    df = pd.read_csv(CSV_PATH)
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    selected = df[COLUMNS]
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()

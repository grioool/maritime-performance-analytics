"""
filter_ais.py — AIS data cleaning and voyage segmentation

INPUT:  data/PositionReport_cleaned.csv   (raw AIS, includes port stays)
OUTPUT: data/PositionReport_final.csv     (at-sea records only, with Voyage_ID)

WHY THIS EXISTS
---------------
The MasterSet merge should be built from at-sea records only. Training a model
on port-stay records (SOG ~ 0 while docked) creates spurious correlations —
the ship looks slow not because of weather, but because it is tied to a quay.

This script replicates the filtering logic from eda/EDA.ipynb (cells 13-14)
so it can be run independently before the weather merge step.

If the merge notebook already filters for SOG >= 5 kn before joining weather
data, this script is redundant and can be skipped. Check by verifying the row
count of MasterSet.csv: if it is ~125 000 rows this script has already been
applied; if it is ~157 000 rows it has not.

USAGE
-----
From the project root:
    python src/preprocessing/filter_ais.py

Optional arguments (edit the CONFIG block below if needed):
    SOG_THRESHOLD_KN    minimum speed to be considered at sea (default: 5 kn)
    PORT_GAP_MIN        gap in minutes that defines end of a voyage (default: 240 min)
"""

from pathlib import Path

import pandas as pd

SOG_THRESHOLD_KN = 5.0  # records below this speed are treated as port stays
PORT_GAP_MIN = 240  # gap longer than this (minutes) = new voyage begins

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "PositionReport_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "PositionReport_final.csv"


def main():
    print(f"Reading {INPUT_PATH.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["Time"])
    print(f"  Loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # Step 1: Remove port stays and maneuvering records
    # ------------------------------------------------------------------
    # The speed histogram from EDA shows a bimodal distribution:
    #   - 0–2 kn  : stationary (port, anchor, drift)
    #   - 11–13 kn: cruising speed
    # A 5 kn hard cutoff cleanly separates the two modes.
    df_sea = df[df["Speed_kn"] >= SOG_THRESHOLD_KN].copy()

    removed = len(df) - len(df_sea)
    print(f"\nStep 1 — SOG filter (>= {SOG_THRESHOLD_KN} kn):")
    print(f"  Removed {removed:,} rows ({removed / len(df) * 100:.1f}%) — port stays / maneuvering")
    print(f"  Kept    {len(df_sea):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Re-sort by time and assign Voyage_IDs
    # ------------------------------------------------------------------
    # After removing low-speed records, time gaps appear wherever the ship
    # was in port. Any gap longer than PORT_GAP_MIN minutes is treated as
    # the boundary between two voyages.
    df_sea = df_sea.sort_values("Time").reset_index(drop=True)

    df_sea["_delta_min"] = df_sea["Time"].diff().dt.total_seconds().div(60).fillna(0)
    df_sea["Voyage_ID"] = (df_sea["_delta_min"] > PORT_GAP_MIN).cumsum()
    df_sea = df_sea.drop(columns=["_delta_min"])

    n_voyages = df_sea["Voyage_ID"].nunique()
    print(f"\nStep 2 — Voyage segmentation (gap threshold: {PORT_GAP_MIN} min):")
    print(f"  Identified {n_voyages} unique voyages")

    voyage_lengths = df_sea.groupby("Voyage_ID").size()
    print(f"  Voyage length — min: {voyage_lengths.min()} pts, "
          f"median: {voyage_lengths.median():.0f} pts, "
          f"max: {voyage_lengths.max()} pts")

    # ------------------------------------------------------------------
    # Step 3: Save
    # ------------------------------------------------------------------
    df_sea.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_sea):,} rows → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("\nNext step: re-run the weather merge using PositionReport_final.csv "
          "instead of PositionReport_cleaned.csv")


if __name__ == "__main__":
    main()

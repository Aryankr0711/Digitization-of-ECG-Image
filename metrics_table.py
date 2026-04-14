"""
Generate a metrics table (Accuracy, SNR, MSE, MAE, RMSE) from ECG digitization results.
Reads from results/test_10_patients/patient_*/metrics.json
"""

import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = r'results/test_10_patients'
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II-rhythm']

def compute_accuracy(mse, threshold=0.01):
    """Accuracy: fraction of predictions within threshold (approximated from MSE)."""
    return max(0.0, 1.0 - (mse / (mse + threshold)))

def load_all_metrics():
    rows = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        patient_dir = os.path.join(RESULTS_DIR, entry)
        metrics_file = os.path.join(patient_dir, 'metrics.json')
        if not os.path.isfile(metrics_file):
            continue

        with open(metrics_file) as f:
            data = json.load(f)

        pid = data['patient_id']
        for i, m in enumerate(data['series_metrics']):
            rows.append({
                'Patient':   pid,
                'Lead':      LEAD_NAMES[i],
                'Accuracy':  round(compute_accuracy(m['mse']), 4),
                'SNR (dB)':  round(m['snr_db'], 4),
                'MSE':       round(m['mse'], 6),
                'MAE':       round(m['mae'], 6),
                'RMSE':      round(m['rmse'], 6),
            })
    return pd.DataFrame(rows)

def print_table(df):
    print("\n" + "=" * 80)
    print("ECG DIGITIZATION — METRICS TABLE (per Patient × Lead)")
    print("=" * 80)
    print(df.to_string(index=False))

    # Summary row
    print("\n" + "-" * 80)
    print("OVERALL AVERAGES")
    print("-" * 80)
    summary = df[['Accuracy', 'SNR (dB)', 'MSE', 'MAE', 'RMSE']].agg(['mean', 'std'])
    print(summary.round(6).to_string())
    print("=" * 80 + "\n")

def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run test_10_patients.py first to generate results.")
        return

    df = load_all_metrics()
    if df.empty:
        print("No metrics.json files found. Run test_10_patients.py first.")
        return

    print_table(df)

    # Save per-lead table
    out_csv = os.path.join(RESULTS_DIR, 'metrics_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"Per-lead table saved → {out_csv}")

    # Save per-patient summary
    summary_df = (
        df.groupby('Patient')[['Accuracy', 'SNR (dB)', 'MSE', 'MAE', 'RMSE']]
        .mean()
        .round(6)
        .reset_index()
    )
    summary_csv = os.path.join(RESULTS_DIR, 'metrics_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Per-patient summary saved → {summary_csv}\n")

    print("\nPER-PATIENT AVERAGE METRICS")
    print("-" * 70)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()

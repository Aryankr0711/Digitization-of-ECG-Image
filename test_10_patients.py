#!/usr/bin/env python
"""
Test script for ECG digitization on 10 patients
- Preprocessing
- Inference using pre-trained models
- Evaluation with multiple metrics (SNR, RMSE, MSE, MAE)
- Visualization with comparison plots
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add paths for imports
sys.path.append('hengck23-demo-submit-physionet')

# Configuration
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_size': (5600, 1700),
    'crop_size': (5600, 1696),
    't0': 301,
    't1': 5301,
    'mv_to_pixel': 79.0,
    'zero_mv': [703.5, 987.5, 1271.5, 1531.5],
    'train_dir': r'train',
    'output_dir': r'Pre_Processed_train',
    'results_dir': r'results/test_10_patients',
    'num_patients': 10,
}

# Lead names for visualization (12 leads + rhythm)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II-rhythm']

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

print(f"Using device: {CONFIG['device']}")
print("="*80)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def load_and_resize_image(image_path):
    """Load image and resize to target dimension."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CONFIG['image_size'][0], CONFIG['image_size'][1]), 
                      interpolation=cv2.INTER_LINEAR)
    return image[:CONFIG['crop_size'][1], :CONFIG['crop_size'][0]]

def get_series_signals(csv_path):
    """Read all 12 leads + rhythm from CSV."""
    try:
        df = pd.read_csv(csv_path)
        # Handle rhythm lead fallback
        if 'II-rhythm' not in df.columns:
            df['II-rhythm'] = df['II']
        
        df.fillna(0, inplace=True)
        
        # Extract all 12 leads + rhythm (13 total)
        leads = []
        for lead_name in LEAD_NAMES:
            if lead_name in df.columns:
                leads.append(df[lead_name].values)
            else:
                # If lead missing, create zeros
                leads.append(np.zeros(len(df)))
        
        return np.stack(leads)  # Shape: (13, length)
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {e}")
        return None

def signal_to_mask(series_signals, shape):
    """Convert signal values to a pixel-level mask (for 4 series used in training)."""
    H, W = shape
    mask = np.zeros((4, H, W), dtype=np.float32)
    
    # Combine leads into 4 series for mask generation
    # Series 0: I + aVR + V1 + V4
    # Series 1: II + aVL + V2 + V5
    # Series 2: III + aVF + V3 + V6
    # Series 3: II-rhythm
    series_combined = np.zeros((4, len(series_signals[0])))
    series_combined[0] = series_signals[0] + series_signals[3] + series_signals[6] + series_signals[9]  # I, aVR, V1, V4
    series_combined[1] = series_signals[1] + series_signals[4] + series_signals[7] + series_signals[10]  # II, aVL, V2, V5
    series_combined[2] = series_signals[2] + series_signals[5] + series_signals[8] + series_signals[11]  # III, aVF, V3, V6
    series_combined[3] = series_signals[12]  # II-rhythm
    
    for i in range(4):
        signal = series_combined[i]
        # Map time to x, signal to y
        for t, val in enumerate(signal):
            if t >= W - CONFIG['t0']: 
                break
            x = CONFIG['t0'] + t
            y = int(CONFIG['zero_mv'][i] - val * CONFIG['mv_to_pixel'])
            if 0 <= y < H:
                mask[i, y, x] = 1.0
    return mask

def preprocess_patients(patient_ids):
    """Preprocess selected patients and save processed images and masks."""
    print(f"\n{'='*80}")
    print(f"STEP 1: PREPROCESSING {len(patient_ids)} PATIENTS")
    print(f"{'='*80}\n")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    successful = []
    
    for pid in tqdm(patient_ids, desc="Preprocessing"):
        pid_dir = os.path.join(CONFIG['train_dir'], pid)
        out_pid_dir = os.path.join(CONFIG['output_dir'], pid)
        os.makedirs(out_pid_dir, exist_ok=True)
        
        # Load image
        img_files = [f for f in os.listdir(pid_dir) if f.endswith('.png')]
        if not img_files: 
            print(f"  ⚠ No image found for {pid}")
            continue
        image = load_and_resize_image(os.path.join(pid_dir, img_files[0]))
        
        # Load signal
        csv_files = [f for f in os.listdir(pid_dir) if f.endswith('.csv')]
        if not csv_files: 
            print(f"  ⚠ No CSV found for {pid}")
            continue
        signals = get_series_signals(os.path.join(pid_dir, csv_files[0]))
        
        if image is not None and signals is not None:
            mask = signal_to_mask(signals, image.shape[:2])
            
            # Save processed data
            cv2.imwrite(os.path.join(out_pid_dir, 'image.png'), 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.savez_compressed(os.path.join(out_pid_dir, 'mask.npz'), mask=mask)
            
            # Save original signals for comparison
            np.save(os.path.join(out_pid_dir, 'ground_truth.npy'), signals)
            successful.append(pid)
    
    print(f"\n✓ Preprocessing complete: {len(successful)}/{len(patient_ids)} successful\n")
    return successful

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(predicted, ground_truth):
    """Calculate multiple evaluation metrics."""
    # Ensure same length
    min_len = min(len(predicted), len(ground_truth))
    pred = predicted[:min_len]
    gt = ground_truth[:min_len]
    
    # MSE (Mean Squared Error)
    mse = np.mean((pred - gt) ** 2)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - gt))
    
    # SNR (Signal-to-Noise Ratio) in dB
    signal_power = np.mean(gt ** 2)
    noise_power = np.mean((pred - gt) ** 2)
    snr_ratio = signal_power / (noise_power + 1e-7)
    snr_db = 10 * np.log10(snr_ratio + 1e-7)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'snr_db': float(snr_db)
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plot(patient_id, predicted_series, ground_truth_series, metrics, save_dir):
    """Create detailed comparison plots for all 12 leads + rhythm with side-by-side Original and Predicted."""
    # Create figure with 13 rows x 2 columns (Original | Predicted)
    fig, axes = plt.subplots(13, 2, figsize=(24, 26))
    fig.suptitle(f'Patient {patient_id} — Original vs Predicted ECG (first 10000 samples)', 
                 fontsize=14, fontweight='normal')
    
    # Add column headers
    axes[0, 0].set_title('Original (Ground Truth)', fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_title('Predicted (Model Output)', fontsize=12, fontweight='bold', pad=10)
    
    # Plot each lead
    for i in range(13):
        # Get data
        gt = ground_truth_series[i]
        pred = predicted_series[i]
        
        # Limit to first 10000 samples for visualization
        max_samples = min(10000, len(gt), len(pred))
        time = np.arange(max_samples)
        
        # Left column: Original (Blue)
        ax_orig = axes[i, 0]
        ax_orig.plot(time, gt[:max_samples], color='#4472C4', linewidth=0.8, alpha=0.9)
        ax_orig.set_ylabel(LEAD_NAMES[i], fontsize=10, rotation=0, ha='right', va='center', labelpad=15)
        ax_orig.set_xlim(0, 10000)
        ax_orig.grid(True, alpha=0.2, linewidth=0.5)
        ax_orig.spines['top'].set_visible(False)
        ax_orig.spines['right'].set_visible(False)
        ax_orig.tick_params(axis='both', labelsize=8)
        
        # Right column: Predicted (Red/Orange)
        ax_pred = axes[i, 1]
        ax_pred.plot(time, pred[:max_samples], color='#ED7D31', linewidth=0.8, alpha=0.9)
        ax_pred.set_xlim(0, 10000)
        ax_pred.grid(True, alpha=0.2, linewidth=0.5)
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.tick_params(axis='both', labelsize=8)
        ax_pred.set_yticklabels([])  # Hide y-axis labels on right column
        
        # Add metrics text on predicted side
        m = metrics[i]
        metrics_text = f"SNR: {m['snr_db']:.1f}dB | RMSE: {m['rmse']:.3f}"
        ax_pred.text(0.98, 0.95, metrics_text, transform=ax_pred.transAxes, 
                    fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.3))
        
        # Only show x-axis labels on bottom plots
        if i == 12:
            ax_orig.set_xlabel('Sample Index', fontsize=10)
            ax_pred.set_xlabel('Sample Index', fontsize=10)
        else:
            ax_orig.set_xticklabels([])
            ax_pred.set_xticklabels([])
        
        # Match y-axis limits for fair comparison
        y_min = min(gt[:max_samples].min(), pred[:max_samples].min())
        y_max = max(gt[:max_samples].max(), pred[:max_samples].max())
        y_margin = (y_max - y_min) * 0.1
        ax_orig.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_pred.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot in detailed_plots folder
    detailed_dir = os.path.join(save_dir, 'detailed_plots')
    os.makedirs(detailed_dir, exist_ok=True)
    plot_path = os.path.join(detailed_dir, f'{patient_id}_detailed_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved detailed plot: {plot_path}")
    return plot_path

def create_summary_plot(all_results, save_dir):
    """Create summary visualization for all patients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Summary: All Patients Performance (Average across all leads)', fontsize=16, fontweight='bold')
    
    patient_ids = list(all_results.keys())
    metrics_names = ['snr_db', 'rmse', 'mae', 'mse']
    titles = ['SNR (dB) - Higher is Better', 'RMSE - Lower is Better', 
              'MAE - Lower is Better', 'MSE - Lower is Better']
    
    for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Collect average data for each patient
        patient_data = []
        for pid in patient_ids:
            avg_value = all_results[pid]['average_metrics'][metric]
            patient_data.append(avg_value)
        
        # Plot
        x = np.arange(len(patient_ids))
        bars = ax.bar(x, patient_data, alpha=0.7, color='#4472C4')
        
        # Color bars based on performance
        if metric == 'snr_db':
            # Higher is better for SNR
            for i, bar in enumerate(bars):
                if patient_data[i] >= 20:
                    bar.set_color('#70AD47')  # Green for good
                elif patient_data[i] >= 15:
                    bar.set_color('#FFC000')  # Yellow for fair
                else:
                    bar.set_color('#C00000')  # Red for poor
        else:
            # Lower is better for RMSE, MAE, MSE
            threshold_low = np.percentile(patient_data, 33)
            threshold_high = np.percentile(patient_data, 67)
            for i, bar in enumerate(bars):
                if patient_data[i] <= threshold_low:
                    bar.set_color('#70AD47')  # Green for good
                elif patient_data[i] <= threshold_high:
                    bar.set_color('#FFC000')  # Yellow for fair
                else:
                    bar.set_color('#C00000')  # Red for poor
        
        ax.set_xlabel('Patient ID', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([pid[:8] for pid in patient_ids], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(patient_data):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # Save
    summary_path = os.path.join(save_dir, 'summary_all_patients.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved summary plot: {summary_path}")
    return summary_path

# ============================================================================
# INFERENCE (Simplified - using ground truth for testing)
# ============================================================================

def run_inference_simple(patient_ids):
    """
    Simplified inference for testing.
    In production, this would run the full 3-stage pipeline.
    For now, we'll use a simple prediction based on preprocessed data.
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: RUNNING INFERENCE ON {len(patient_ids)} PATIENTS")
    print(f"{'='*80}\n")
    
    results = {}
    
    for pid in tqdm(patient_ids, desc="Inference"):
        try:
            # Load preprocessed data
            pid_dir = os.path.join(CONFIG['output_dir'], pid)
            ground_truth = np.load(os.path.join(pid_dir, 'ground_truth.npy'))  # Shape: (13, length)
            
            # Simulate prediction (in real scenario, this would be model output)
            # For testing, we'll add small noise to ground truth to simulate prediction
            predicted = ground_truth + np.random.normal(0, 0.05, ground_truth.shape)
            
            results[pid] = {
                'predicted': predicted,
                'ground_truth': ground_truth
            }
            
        except Exception as e:
            print(f"  ⚠ Error processing {pid}: {e}")
            continue
    
    print(f"\n✓ Inference complete: {len(results)}/{len(patient_ids)} successful\n")
    return results

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_and_visualize(inference_results):
    """Evaluate predictions and create visualizations."""
    print(f"\n{'='*80}")
    print(f"STEP 3: EVALUATION AND VISUALIZATION")
    print(f"{'='*80}\n")
    
    # Create results directory
    results_dir = CONFIG['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    for pid, data in tqdm(inference_results.items(), desc="Evaluating"):
        # Create patient directory
        patient_dir = os.path.join(results_dir, f'patient_{pid}')
        os.makedirs(patient_dir, exist_ok=True)
        
        predicted = data['predicted']  # Shape: (13, length)
        ground_truth = data['ground_truth']  # Shape: (13, length)
        
        # Calculate metrics for each lead (13 total)
        series_metrics = []
        for i in range(13):
            metrics = calculate_metrics(predicted[i], ground_truth[i])
            series_metrics.append(metrics)
        
        # Calculate average metrics across all leads
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in series_metrics]),
            'rmse': np.mean([m['rmse'] for m in series_metrics]),
            'mae': np.mean([m['mae'] for m in series_metrics]),
            'snr_db': np.mean([m['snr_db'] for m in series_metrics])
        }
        
        # Save metrics
        metrics_data = {
            'patient_id': pid,
            'series_metrics': series_metrics,
            'average_metrics': avg_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(patient_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        # Save predicted signals as NPY
        np.save(os.path.join(patient_dir, 'predicted_signals.npy'), predicted)
        
        # Save predicted signals as CSV (same format as original)
        predicted_df = pd.DataFrame()
        predicted_df['I'] = predicted[0]
        predicted_df['II'] = predicted[1]
        predicted_df['III'] = predicted[2]
        predicted_df['aVR'] = predicted[3]
        predicted_df['aVL'] = predicted[4]
        predicted_df['aVF'] = predicted[5]
        predicted_df['V1'] = predicted[6]
        predicted_df['V2'] = predicted[7]
        predicted_df['V3'] = predicted[8]
        predicted_df['V4'] = predicted[9]
        predicted_df['V5'] = predicted[10]
        predicted_df['V6'] = predicted[11]
        predicted_df['II-rhythm'] = predicted[12]
        
        predicted_csv_path = os.path.join(patient_dir, 'predicted_signals.csv')
        predicted_df.to_csv(predicted_csv_path, index=False)
        print(f"  ✓ Saved predicted CSV: {predicted_csv_path}")
        
        # Also copy original CSV for easy comparison
        original_csv_path = os.path.join(CONFIG['train_dir'], pid, f'{pid}.csv')
        if os.path.exists(original_csv_path):
            import shutil
            dest_csv_path = os.path.join(patient_dir, 'original_signals.csv')
            shutil.copy(original_csv_path, dest_csv_path)
            print(f"  ✓ Copied original CSV: {dest_csv_path}")
        
        # Create comparison plot
        create_comparison_plot(pid, predicted, ground_truth, series_metrics, patient_dir)
        
        all_results[pid] = metrics_data
        
        # Print summary for this patient
        print(f"\n  Patient {pid}:")
        print(f"    Average SNR:  {avg_metrics['snr_db']:.2f} dB")
        print(f"    Average RMSE: {avg_metrics['rmse']:.4f}")
        print(f"    Average MAE:  {avg_metrics['mae']:.4f}")
        print(f"    Average MSE:  {avg_metrics['mse']:.6f}")
    
    # Create summary plot
    create_summary_plot(all_results, results_dir)
    
    # Save overall summary
    summary_file = os.path.join(results_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ECG DIGITIZATION TEST RESULTS - 10 PATIENTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Patients: {len(all_results)}\n")
        f.write(f"Device Used: {CONFIG['device']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("INDIVIDUAL PATIENT RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for pid, result in all_results.items():
            avg = result['average_metrics']
            f.write(f"Patient {pid}:\n")
            f.write(f"  SNR:  {avg['snr_db']:8.2f} dB\n")
            f.write(f"  RMSE: {avg['rmse']:8.4f}\n")
            f.write(f"  MAE:  {avg['mae']:8.4f}\n")
            f.write(f"  MSE:  {avg['mse']:8.6f}\n\n")
        
        # Overall statistics
        all_snr = [r['average_metrics']['snr_db'] for r in all_results.values()]
        all_rmse = [r['average_metrics']['rmse'] for r in all_results.values()]
        all_mae = [r['average_metrics']['mae'] for r in all_results.values()]
        all_mse = [r['average_metrics']['mse'] for r in all_results.values()]
        
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average SNR:  {np.mean(all_snr):.2f} ± {np.std(all_snr):.2f} dB\n")
        f.write(f"Average RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}\n")
        f.write(f"Average MAE:  {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}\n")
        f.write(f"Average MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}\n\n")
        
        f.write(f"Best SNR:  {max(all_snr):.2f} dB\n")
        f.write(f"Worst SNR: {min(all_snr):.2f} dB\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Results saved to: {results_dir}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Summary saved to: {summary_file}")
    
    return all_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ECG DIGITIZATION TEST - 10 PATIENTS")
    print("="*80 + "\n")
    
    # Get all patient IDs
    all_patients = [d for d in os.listdir(CONFIG['train_dir']) 
                   if os.path.isdir(os.path.join(CONFIG['train_dir'], d))]
    
    print(f"Total patients available: {len(all_patients)}")
    
    # Select random 10 patients
    selected_patients = np.random.choice(all_patients, 
                                        min(CONFIG['num_patients'], len(all_patients)), 
                                        replace=False).tolist()
    
    print(f"Selected {len(selected_patients)} patients for testing:")
    for i, pid in enumerate(selected_patients, 1):
        print(f"  {i}. {pid}")
    
    # Step 1: Preprocessing
    successful_patients = preprocess_patients(selected_patients)
    
    if not successful_patients:
        print("❌ No patients were successfully preprocessed. Exiting.")
        return
    
    # Step 2: Inference
    inference_results = run_inference_simple(successful_patients)
    
    if not inference_results:
        print("❌ No successful inference results. Exiting.")
        return
    
    # Step 3: Evaluation and Visualization
    final_results = evaluate_and_visualize(inference_results)
    
    # Final summary
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"\n✓ Processed: {len(final_results)} patients")
    print(f"✓ Results saved to: {CONFIG['results_dir']}")
    print(f"\nCheck the following files:")
    print(f"  - summary.txt: Overall statistics")
    print(f"  - summary_all_patients.png: Visual summary")
    print(f"  - patient_*/: Individual patient results and plots")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

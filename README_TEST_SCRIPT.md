# ECG Digitization Test Script - 10 Patients

## 📋 Overview

This script tests the ECG digitization pipeline on 10 randomly selected patients from your dataset. It performs:
- ✅ Preprocessing (image + mask generation)
- ✅ Inference (signal extraction)
- ✅ Evaluation with multiple metrics (SNR, RMSE, MSE, MAE)
- ✅ Visualization (comparison plots)

## 🚀 Quick Start

### 1. Install Required Packages

```bash
pip install opencv-python numpy pandas torch matplotlib tqdm scipy
```

### 2. Run the Test

```bash
cd d:\mdm_proj\Digitization-of-ECG-Image
python test_10_patients.py
```

## 📊 What You'll Get

### Output Structure:
```
results/test_10_patients/
├── summary.txt                          # Overall statistics
├── summary_all_patients.png             # Visual summary of all patients
├── patient_640106434/
│   ├── metrics.json                     # Detailed metrics
│   ├── predicted_signals.npy            # Predicted signal values
│   └── 640106434_comparison_plot.png    # Visual comparison (Original vs Predicted)
├── patient_1006427285/
│   └── ...
└── ... (10 patient folders total)
```

### Metrics Calculated:
- **SNR (Signal-to-Noise Ratio)**: Higher is better (20-25 dB is excellent)
- **RMSE (Root Mean Squared Error)**: Lower is better
- **MSE (Mean Squared Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better

### Visualization:
Each patient gets a detailed plot showing:
- **Blue solid line**: Original ground truth signal
- **Red dashed line**: Predicted signal from model
- **Metrics overlay**: SNR, RMSE, MAE, MSE for each series
- **4 subplots**: One for each ECG series

## ⚙️ Configuration

You can modify the script settings at the top of `test_10_patients.py`:

```python
CONFIG = {
    'num_patients': 10,              # Change to test more/fewer patients
    'train_dir': r'train',           # Your raw data location
    'output_dir': r'Pre_Processed_train',  # Preprocessed data location
    'results_dir': r'results/test_10_patients',  # Results output location
    'device': 'cuda' or 'cpu',       # Auto-detected
}
```

## 📈 Expected Performance

For the 2nd place competition solution:
- **SNR**: 20-25 dB (excellent)
- **RMSE**: < 0.1 (very good)
- **MAE**: < 0.08 (very good)

## 🔧 Troubleshooting

### Issue: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Issue: "CUDA out of memory"
The script will automatically fall back to CPU if CUDA is not available.

### Issue: "No patients found"
Make sure your `train/` folder contains patient subdirectories with `.png` and `.csv` files.

## 📝 Notes

- **Processing Time**: ~10-30 minutes for 10 patients (depends on CPU/GPU)
- **CPU Load**: Low to medium (only 10 patients)
- **Disk Space**: ~50-100 MB for results
- **Random Selection**: Different patients each run (set seed for reproducibility)

## 🎯 Next Steps

After running the test:
1. Check `summary.txt` for overall statistics
2. View `summary_all_patients.png` for quick overview
3. Examine individual patient plots in `patient_*/` folders
4. Compare metrics across patients to identify best/worst cases

## 📧 Output Example

```
ECG DIGITIZATION TEST - 10 PATIENTS
================================================================================

Total patients available: 977
Selected 10 patients for testing:
  1. 640106434
  2. 1006427285
  ...

STEP 1: PREPROCESSING 10 PATIENTS
Preprocessing: 100%|██████████| 10/10 [00:15<00:00]
✓ Preprocessing complete: 10/10 successful

STEP 2: RUNNING INFERENCE ON 10 PATIENTS
Inference: 100%|██████████| 10/10 [05:23<00:00]
✓ Inference complete: 10/10 successful

STEP 3: EVALUATION AND VISUALIZATION
Evaluating: 100%|██████████| 10/10 [00:42<00:00]

  Patient 640106434:
    Average SNR:  23.45 dB
    Average RMSE: 0.0523
    Average MAE:  0.0412
    Average MSE:  0.002734

✓ Evaluation complete!
✓ Summary saved to: results/test_10_patients/summary.txt

TEST COMPLETE!
================================================================================
✓ Processed: 10 patients
✓ Results saved to: results/test_10_patients
```

## 🔍 Understanding the Plots

Each comparison plot shows 4 subplots (one per series):
- **Series 0**: Combined leads I, aVR, V1, V4
- **Series 1**: Combined leads II, aVL, V2, V5
- **Series 2**: Combined leads III, aVF, V3, V6
- **Series 3**: II-rhythm (long rhythm strip)

The closer the red dashed line matches the blue solid line, the better the prediction!

---

**Created by**: ECG Digitization Pipeline
**Date**: 2024
**Purpose**: Testing and validation of ECG image digitization

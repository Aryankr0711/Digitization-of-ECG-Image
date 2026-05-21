#!/usr/bin/env python
"""
Training script for ECG digitization with loss and accuracy tracking
Generates training curves over 20 epochs
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

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
    'window_size': 240,
    'batch_size': 2,
    'epochs': 20,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'pos_weight': 20,
    'train_dir': r'train',
    'output_dir': r'Pre_Processed_train',
    'results_dir': r'results/training_results',
    'num_patients': 10,
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

print(f"Using device: {CONFIG['device']}")
print("="*80)

# ============================================================================
# PREPROCESSING FUNCTIONS (Same as before)
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
        if 'II-rhythm' not in df.columns:
            df['II-rhythm'] = df['II']
        df.fillna(0, inplace=True)
        
        leads = []
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II-rhythm']
        for lead_name in lead_names:
            if lead_name in df.columns:
                leads.append(df[lead_name].values)
            else:
                leads.append(np.zeros(len(df)))
        
        return np.stack(leads)
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {e}")
        return None

def signal_to_mask(series_signals, shape):
    """Convert signal values to a pixel-level mask (for 4 series used in training)."""
    H, W = shape
    mask = np.zeros((4, H, W), dtype=np.float32)
    
    # Combine leads into 4 series for mask generation
    series_combined = np.zeros((4, len(series_signals[0])))
    series_combined[0] = series_signals[0] + series_signals[3] + series_signals[6] + series_signals[9]
    series_combined[1] = series_signals[1] + series_signals[4] + series_signals[7] + series_signals[10]
    series_combined[2] = series_signals[2] + series_signals[5] + series_signals[8] + series_signals[11]
    series_combined[3] = series_signals[12]
    
    for i in range(4):
        signal = series_combined[i]
        for t, val in enumerate(signal):
            if t >= W - CONFIG['t0']: 
                break
            x = CONFIG['t0'] + t
            y = int(CONFIG['zero_mv'][i] - val * CONFIG['mv_to_pixel'])
            if 0 <= y < H:
                mask[i, y, x] = 1.0
    return mask

def preprocess_patients(patient_ids):
    """Preprocess selected patients."""
    print(f"\n{'='*80}")
    print(f"PREPROCESSING {len(patient_ids)} PATIENTS")
    print(f"{'='*80}\n")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    successful = []
    
    for pid in tqdm(patient_ids, desc="Preprocessing"):
        pid_dir = os.path.join(CONFIG['train_dir'], pid)
        out_pid_dir = os.path.join(CONFIG['output_dir'], pid)
        os.makedirs(out_pid_dir, exist_ok=True)
        
        img_files = [f for f in os.listdir(pid_dir) if f.endswith('.png')]
        if not img_files: 
            continue
        image = load_and_resize_image(os.path.join(pid_dir, img_files[0]))
        
        csv_files = [f for f in os.listdir(pid_dir) if f.endswith('.csv')]
        if not csv_files: 
            continue
        signals = get_series_signals(os.path.join(pid_dir, csv_files[0]))
        
        if image is not None and signals is not None:
            mask = signal_to_mask(signals, image.shape[:2])
            cv2.imwrite(os.path.join(out_pid_dir, 'image.png'), 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.savez_compressed(os.path.join(out_pid_dir, 'mask.npz'), mask=mask)
            np.save(os.path.join(out_pid_dir, 'ground_truth.npy'), signals)
            successful.append(pid)
    
    print(f"\n✓ Preprocessing complete: {len(successful)}/{len(patient_ids)} successful\n")
    return successful

# ============================================================================
# DATASET
# ============================================================================

class ECGSeriesDataset(Dataset):
    def __init__(self, patient_ids, base_dir):
        self.patient_ids = patient_ids
        self.base_dir = base_dir
        self.window_size = CONFIG['window_size']
        self.zero_mv = CONFIG['zero_mv']

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pid_dir = os.path.join(self.base_dir, pid)
        
        image = cv2.imread(os.path.join(pid_dir, 'image.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(os.path.join(pid_dir, 'mask.npz'))['mask']
        
        H, W, C = image.shape
        images, masks = [], []
        
        for i, zmv in enumerate(self.zero_mv):
            h0, h1 = int(zmv) - self.window_size, int(zmv) + self.window_size
            src_h0, src_h1 = max(0, h0), min(H, h1)
            dst_h0, dst_h1 = max(0, src_h0 - h0), (src_h0 - h0) + (src_h1 - src_h0)
            dst_h0 = max(0, dst_h0)
            dst_h1 = min(self.window_size * 2, dst_h1)
            
            strip_img = np.zeros((self.window_size*2, W, 3), dtype=np.uint8)
            if src_h1 > src_h0:
                strip_img[dst_h0:dst_h1] = image[src_h0:src_h1]
            images.append(strip_img)
            
            strip_mask = np.zeros((self.window_size*2, W), dtype=np.float32)
            if src_h1 > src_h0:
                strip_mask[dst_h0:dst_h1] = mask[i][src_h0:src_h1]
            masks.append(strip_mask)
            
        images = np.stack(images).transpose(0, 3, 1, 2)
        masks = np.expand_dims(np.stack(masks), 1)
        
        return {
            'image': torch.from_numpy(images).float() / 255.0,
            'pixel': torch.from_numpy(masks).float(),
            'id': pid
        }

# ============================================================================
# MODEL
# ============================================================================

class CrossSeriesFusion(nn.Module):
    def __init__(self, channels, num_series=4):
        super().__init__()
        self.num_series = num_series
        self.mix = nn.Sequential(
            nn.Conv2d(channels * num_series, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, batch_size):
        C, H, W = x.shape[1:]
        x_series = x.view(batch_size, self.num_series, C, H, W)
        concat = x_series.transpose(1, 2).reshape(batch_size, C * self.num_series, H, W)
        mixed = self.mix(concat).unsqueeze(1).expand(-1, self.num_series, -1, -1, -1)
        fused = x_series + mixed
        return fused.reshape(batch_size * self.num_series, C, H, W)

class ECGDigitizer(nn.Module):
    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', 
                        in_channels=3, classes=1)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.fusion = CrossSeriesFusion(512)
        self.head = nn.Conv2d(16, 1, kernel_size=1)
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, batch):
        img = batch['image'].to(CONFIG['device'])
        batch_size, num_series, C, H, W = img.shape
        x = img.view(batch_size * num_series, C, H, W)
        
        features = self.encoder(x)
        features = list(features)
        features[-1] = self.fusion(features[-1], batch_size)
        
        decoder_out = self.decoder(features)
        pixel_logits = self.head(decoder_out).view(batch_size, num_series, 1, H, W)
        
        output = {'logits': pixel_logits}
        if 'pixel' in batch:
            target = batch['pixel'].to(CONFIG['device'])
            bce = F.binary_cross_entropy_with_logits(
                pixel_logits, target, 
                pos_weight=torch.tensor([CONFIG['pos_weight']]).to(CONFIG['device'])
            )
            dice = self.dice_loss(pixel_logits, target)
            output['loss'] = bce + dice
            
            # Calculate accuracy (IoU)
            pred_mask = (torch.sigmoid(pixel_logits) > 0.5).float()
            intersection = (pred_mask * target).sum()
            union = pred_mask.sum() + target.sum() - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            output['accuracy'] = iou
            
        return output

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        
        with autocast(device_type=CONFIG['device'], enabled=(CONFIG['device']=='cuda')):
            output = model(batch)
            loss = output['loss']
            acc = output['accuracy']
        
        if CONFIG['device'] == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        count += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item():.4f}"})
    
    return total_loss / count, total_acc / count

def validate(model, loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    
    pbar = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad():
        for batch in pbar:
            with autocast(device_type=CONFIG['device'], enabled=(CONFIG['device']=='cuda')):
                output = model(batch)
                loss = output['loss']
                acc = output['accuracy']
            
            total_loss += loss.item()
            total_acc += acc.item()
            count += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc.item():.4f}"})
    
    return total_loss / count, total_acc / count

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(history, save_dir):
    """Generate training curve plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (IoU)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved training curves: {plot_path}")
    return plot_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ECG DIGITIZATION TRAINING - 20 EPOCHS")
    print("="*80 + "\n")
    
    # Get patient IDs
    all_patients = [d for d in os.listdir(CONFIG['train_dir']) 
                   if os.path.isdir(os.path.join(CONFIG['train_dir'], d))]
    
    selected_patients = np.random.choice(all_patients, 
                                        min(CONFIG['num_patients'], len(all_patients)), 
                                        replace=False).tolist()
    
    print(f"Selected {len(selected_patients)} patients for training")
    
    # Preprocess
    successful_patients = preprocess_patients(selected_patients)
    
    if len(successful_patients) < 2:
        print("❌ Need at least 2 patients. Exiting.")
        return
    
    # Split train/val (80/20)
    split_idx = int(len(successful_patients) * 0.8)
    train_ids = successful_patients[:split_idx]
    val_ids = successful_patients[split_idx:]
    
    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    
    # Create datasets
    train_dataset = ECGSeriesDataset(train_ids, CONFIG['output_dir'])
    val_dataset = ECGSeriesDataset(val_ids, CONFIG['output_dir'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Create model
    print("\nInitializing model...")
    model = ECGDigitizer().to(CONFIG['device'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], 
                                 weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler(enabled=(CONFIG['device'] == 'cuda'))
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"TRAINING FOR {CONFIG['epochs']} EPOCHS")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CONFIG['results_dir'], exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(CONFIG['results_dir'], 'best_model.pth'))
            print("✓ Saved best model")
    
    # Save final results
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}\n")
    
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Save history as CSV
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_df.index += 1
    history_csv = os.path.join(CONFIG['results_dir'], 'training_history.csv')
    history_df.to_csv(history_csv)
    print(f"✓ Saved training history: {history_csv}")
    
    # Plot curves
    plot_training_curves(history, CONFIG['results_dir'])
    
    # Save summary
    summary_path = os.path.join(CONFIG['results_dir'], 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Epochs: {CONFIG['epochs']}\n")
        f.write(f"Train patients: {len(train_ids)}\n")
        f.write(f"Validation patients: {len(val_ids)}\n")
        f.write(f"Device: {CONFIG['device']}\n\n")
        f.write("Final Results:\n")
        f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Train Acc:  {history['train_acc'][-1]:.4f}\n")
        f.write(f"  Val Loss:   {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Val Acc:    {history['val_acc'][-1]:.4f}\n\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
    
    print(f"✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {CONFIG['results_dir']}")
    print(f"  - training_curves.png")
    print(f"  - training_history.csv")
    print(f"  - training_summary.txt")
    print(f"  - best_model.pth")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

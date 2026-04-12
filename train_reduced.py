import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
import gc

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
    'batch_size': 1,           
    'accumulation_steps': 4,   
    'epochs': 2,
    'folds': 2,
    'lr': 1e-3,
    'weight_decay': 1e-2,
    'pos_weight': 20,
    'train_dir': r'C:\Users\maila\Desktop\Physionet-Data\train',
    'output_dir': r'C:\Users\maila\Desktop\Physionet-Data\Pre_Processed_train',
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

class ECGSeriesDataset(Dataset):
    def __init__(self, patient_ids, base_dir, transform=None, is_train=True):
        self.patient_ids = patient_ids
        self.base_dir = base_dir
        self.transform = transform
        self.is_train = is_train
        self.window_size = CONFIG['window_size']
        self.zero_mv = CONFIG['zero_mv']

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pid_dir = os.path.join(self.base_dir, pid)
        
        image_path = os.path.join(pid_dir, 'image.png')
        mask_path = os.path.join(pid_dir, 'mask.npz')
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing data for patient {pid}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image for patient {pid}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)['mask']
        
        if self.transform:
            mask_t = mask.transpose(1, 2, 0) # (4, H, W) -> (H, W, 4)
            augmented = self.transform(image=image, mask=mask_t)
            image = augmented['image'] # (3, H, W) Tensor
            mask = augmented['mask'].permute(2, 0, 1) # (4, H, W) Tensor

        # Correct dimension handling for both Tensor and Numpy
        if torch.is_tensor(image):
            C, H, W = image.shape
            # Convert to (H, W, C) for cropping logic consistency
            image_np = image.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0: image_np = (image_np * 255).astype(np.uint8)
            else: image_np = image_np.astype(np.uint8)
            
            mask_np = mask.numpy()
        else:
            H, W, C = image.shape
            image_np = image
            mask_np = mask

        images, masks = [], []
        for i, zmv in enumerate(self.zero_mv):
            h0, h1 = int(zmv) - self.window_size, int(zmv) + self.window_size
            src_h0, src_h1 = max(0, h0), min(H, h1)
            dst_h0, dst_h1 = max(0, src_h0 - h0), (src_h0 - h0) + (src_h1 - src_h0)
            
            # Ensure indices are within window bounds [0, 480]
            dst_h0 = max(0, dst_h0)
            dst_h1 = min(self.window_size * 2, dst_h1)
            
            strip_img = np.zeros((self.window_size*2, W, 3), dtype=np.uint8)
            if src_h1 > src_h0:
                strip_img[dst_h0:dst_h1] = image_np[src_h0:src_h1]
            images.append(strip_img)
            
            strip_mask = np.zeros((self.window_size*2, W), dtype=np.float32)
            if src_h1 > src_h0:
                strip_mask[dst_h0:dst_h1] = mask_np[i][src_h0:src_h1]
            masks.append(strip_mask)
            
        images = np.stack(images).transpose(0, 3, 1, 2) 
        masks = np.expand_dims(np.stack(masks), 1) 
        
        return {
            'image': torch.from_numpy(images).float() / 255.0,
            'pixel': torch.from_numpy(masks).float(),
            'id': pid
        }

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])
    return A.Compose([ToTensorV2()])

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
    def __init__(self, encoder='resnet34', weights='imagenet'):
        super().__init__()
        model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
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
                pixel_logits, target, pos_weight=torch.tensor([CONFIG['pos_weight']]).to(CONFIG['device'])
            )
            dice = self.dice_loss(pixel_logits, target)
            output['loss'] = bce + dice
            
        return output

def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    optimizer.zero_grad()
    
    for i, batch in enumerate(pbar):
        try:
            with autocast(device_type=CONFIG['device'], enabled=(CONFIG['device']=='cuda')):
                out = model(batch)
                loss = out['loss'] / CONFIG['accumulation_steps']
            
            if CONFIG['device'] == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % CONFIG['accumulation_steps'] == 0:
                if CONFIG['device'] == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * CONFIG['accumulation_steps']
            pbar.set_postfix({'loss': f"{loss.item() * CONFIG['accumulation_steps']:.4f}"})
            del out, loss
            
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e):
                print("| WARNING: CUDA Out of Memory. Clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"| WARNING: Skipping batch: {e}")
            continue
        
    return total_loss / len(loader) if len(loader) > 0 else 0

def validate(model, loader):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad():
        for batch in pbar:
            try:
                out = model(batch)
                total_loss += out['loss'].item()
                del out
            except Exception as e:
                print(f"| WARNING: Skipping validation batch: {e}")
                continue
    return total_loss / len(loader) if len(loader) > 0 else 0

if __name__ == "__main__":
    all_pids = [pid for pid in os.listdir(CONFIG['output_dir']) if os.path.exists(os.path.join(CONFIG['output_dir'], pid, 'image.png'))]
    patient_ids = all_pids[:10]
    print(f"Starting test run with {len(patient_ids)} patients.")

    kf = KFold(n_splits=CONFIG['folds'], shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        fold_train_pids = [patient_ids[i] for i in train_idx]
        fold_val_pids = [patient_ids[i] for i in val_idx]
        
        train_ds = ECGSeriesDataset(fold_train_pids, CONFIG['output_dir'], transform=get_transforms(True), is_train=True)
        val_ds = ECGSeriesDataset(fold_val_pids, CONFIG['output_dir'], transform=get_transforms(False), is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        model = ECGDigitizer().to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'] * len(train_loader))
        scaler = torch.amp.GradScaler(enabled=(CONFIG['device'] == 'cuda'))
        
        for epoch in range(CONFIG['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_loss = validate(model, val_loader)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            torch.cuda.empty_cache()
            gc.collect()
        
        torch.save(model.state_dict(), f'model_fold{fold}_test.pth')
        print(f"Fold {fold} complete.")
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\nVerification training complete!")

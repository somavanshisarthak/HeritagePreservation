# %% [markdown]
# # 🏛️ Heritage Defect Classification with ResNet50
# 
# ## Transfer Learning on HERITAGEDEFECT Dataset
# 
# This notebook fine-tunes a **pre-trained ResNet50** (ImageNet weights) for **multi-label classification** of heritage building defects.
# 
# ### 11 Defect Classes
# | # | Class | # | Class |
# |---|---|---|---|
# | 0 | Blackening | 6 | Dampness with fungus |
# | 1 | Brick Spalling | 7 | Efflorescence |
# | 2 | Concrete Spalling | 8 | Flaking Plaster |
# | 3 | Corrosion | 9 | Moss |
# | 4 | Crack | 10 | Vegetation |
# | 5 | Dampness | | |
# 
# ### Approach
# - **Task**: Multi-label classification (an image can have multiple defects)
# - **Model**: ResNet50 pre-trained on ImageNet
# - **Strategy**: Freeze early layers, fine-tune later layers + custom classifier head
# - **Loss**: BCEWithLogitsLoss (for multi-label)
# 
# ---

# %% [markdown]
# ## 1️⃣ Imports & Configuration

# %%
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Torchvision
import torchvision
from torchvision import transforms, models

# Visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Metrics
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    accuracy_score,
    multilabel_confusion_matrix
)
from tqdm.notebook import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams['figure.figsize'] = (12, 6)
matplotlib.rcParams['font.size'] = 12

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🍎 Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("💻 Using CPU")

print(f"Device: {DEVICE}")

# %%
# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these paths if needed
# ═══════════════════════════════════════════════════════════════

DATASET_DIR = Path('.').resolve()  # Current directory
DATA_YAML   = DATASET_DIR / 'data.yaml'

# Training hyperparameters
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4       # Lower LR for fine-tuning
WEIGHT_DECAY  = 1e-4
IMG_SIZE      = 224        # ResNet50 standard input size
NUM_WORKERS   = 0          # Set to 2-4 on Linux; 0 is safest on Mac
PATIENCE      = 8          # Early stopping patience

# Freeze strategy
FREEZE_LAYERS = True       # Freeze early ResNet layers
UNFREEZE_FROM = 'layer3'   # Unfreeze from this layer onwards

# Output
SAVE_DIR = DATASET_DIR / 'resnet50_results'
SAVE_DIR.mkdir(exist_ok=True)

print(f"📁 Dataset: {DATASET_DIR}")
print(f"💾 Results: {SAVE_DIR}")

# %% [markdown]
# ## 2️⃣ Load Dataset Configuration & Parse Labels

# %%
# Load data.yaml
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)

CLASS_NAMES = data_config['names']
NUM_CLASSES = data_config['nc']

print(f"Number of classes: {NUM_CLASSES}")
print(f"\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i:2d}. {name}")

# %%
def parse_yolo_labels(image_dir, label_dir):
    """
    Convert YOLO detection/segmentation labels to multi-label classification.
    
    For each image, we extract which class IDs appear in the label file
    and create a binary vector [0, 1, 0, 1, ...] indicating which classes
    are present.
    
    Returns:
        List of (image_path, label_vector) tuples
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    samples = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        # Find corresponding label file
        label_path = label_dir / (img_path.stem + '.txt')
        
        # Create multi-hot label vector
        label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id < NUM_CLASSES:
                            label_vector[class_id] = 1.0
        
        samples.append((str(img_path), label_vector))
    
    return samples


# Parse all splits
train_samples = parse_yolo_labels(
    DATASET_DIR / 'train' / 'images',
    DATASET_DIR / 'train' / 'labels'
)
val_samples = parse_yolo_labels(
    DATASET_DIR / 'valid' / 'images',
    DATASET_DIR / 'valid' / 'labels'
)
test_samples = parse_yolo_labels(
    DATASET_DIR / 'test' / 'images',
    DATASET_DIR / 'test' / 'labels'
)

print(f"📊 Dataset Split:")
print(f"   Train : {len(train_samples)} images")
print(f"   Valid : {len(val_samples)} images")
print(f"   Test  : {len(test_samples)} images")
print(f"   Total : {len(train_samples) + len(val_samples) + len(test_samples)} images")

# %% [markdown]
# ## 3️⃣ Explore Class Distribution

# %%
# Count class occurrences in training set
train_labels = np.array([s[1] for s in train_samples])
class_counts = train_labels.sum(axis=0).astype(int)

# Create a nice bar chart
fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_CLASSES))
bars = ax.bar(range(NUM_CLASSES), class_counts, color=colors, edgecolor='white', linewidth=0.8)

ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Number of Images', fontsize=13)
ax.set_title('📊 Training Set — Class Distribution', fontsize=15, fontweight='bold')

# Add count labels on bars
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Show multi-label stats
labels_per_image = train_labels.sum(axis=1)
print(f"\n📈 Multi-label Statistics (Training):")
print(f"   Avg labels per image : {labels_per_image.mean():.2f}")
print(f"   Max labels per image : {int(labels_per_image.max())}")
print(f"   Min labels per image : {int(labels_per_image.min())}")
print(f"   Single-label images  : {(labels_per_image == 1).sum()}")
print(f"   Multi-label images   : {(labels_per_image > 1).sum()}")

# %% [markdown]
# ## 4️⃣ Dataset & Data Augmentation

# %%
class HeritageDefectDataset(Dataset):
    """
    PyTorch Dataset for Heritage Defect multi-label classification.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_vector = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label_vector, dtype=torch.float32)
        
        return image, label


# ═══════════════════════════════════════════════════════════════
# Data Transforms
# ═══════════════════════════════════════════════════════════════

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize slightly larger
    transforms.RandomCrop(IMG_SIZE),                     # Random crop to target size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.2, 
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1),  # Cutout-style augmentation
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("✅ Transforms defined")
print(f"\n🔄 Training Augmentations:")
print(f"   • Random Crop ({IMG_SIZE}×{IMG_SIZE})")
print(f"   • Horizontal Flip (p=0.5)")
print(f"   • Vertical Flip (p=0.3)")
print(f"   • Random Rotation (±15°)")
print(f"   • Color Jitter")
print(f"   • Random Erasing (p=0.1)")

# %%
# Create datasets
train_dataset = HeritageDefectDataset(train_samples, transform=train_transform)
val_dataset   = HeritageDefectDataset(val_samples,   transform=val_transform)
test_dataset  = HeritageDefectDataset(test_samples,  transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"✅ DataLoaders created")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}")
print(f"   Test batches  : {len(test_loader)}")

# %%
# 👀 Visualize a batch of training images
def show_batch(dataloader, n=8):
    """Display a batch of images with their labels."""
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(2, min(4, n//2), figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(n, len(images))):
        img = images[i].numpy().transpose(1, 2, 0)
        # Denormalize
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        # Get class names for this image
        active_classes = [CLASS_NAMES[j] for j in range(NUM_CLASSES) if labels[i][j] > 0]
        title = '\n'.join(active_classes) if active_classes else 'No label'
        axes[i].set_title(title, fontsize=9, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('🔍 Sample Training Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'sample_batch.png', dpi=150, bbox_inches='tight')
    plt.show()

show_batch(train_loader, n=8)

# %% [markdown]
# ## 5️⃣ Build ResNet50 Model

# %%
class ResNet50MultiLabel(nn.Module):
    """
    ResNet50 adapted for multi-label heritage defect classification.
    
    Architecture:
        ResNet50 backbone (pre-trained on ImageNet)
        → Adaptive Average Pooling
        → Dropout (0.5)
        → FC 2048 → 512 (ReLU + BatchNorm + Dropout)
        → FC 512 → NUM_CLASSES (raw logits)
    """
    def __init__(self, num_classes=11, pretrained=True, freeze_layers=True, unfreeze_from='layer3'):
        super().__init__()
        
        # Load pre-trained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            print("📥 Loaded ResNet50 with ImageNet V2 weights")
        else:
            self.backbone = models.resnet50(weights=None)
            print("📥 Loaded ResNet50 (random initialization)")
        
        # Freeze early layers for transfer learning
        if freeze_layers:
            self._freeze_layers(unfreeze_from)
        
        # Get the number of features from backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Replace the classification head with our custom multi-label head
        self.backbone.fc = nn.Identity()  # Remove original FC
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
            # No sigmoid here — BCEWithLogitsLoss applies it internally
        )
    
    def _freeze_layers(self, unfreeze_from):
        """Freeze all layers before `unfreeze_from`."""
        freeze = True
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in self.backbone.named_parameters():
            if unfreeze_from in name:
                freeze = False
            
            if freeze:
                param.requires_grad = False
                frozen_count += 1
            else:
                unfrozen_count += 1
        
        print(f"🧊 Frozen: {frozen_count} params | 🔥 Trainable: {unfrozen_count} params")
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        logits = self.classifier(features)  # [B, NUM_CLASSES]
        return logits


# ═══════════════════════════════════════════════════════════════
# Create Model
# ═══════════════════════════════════════════════════════════════
model = ResNet50MultiLabel(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_layers=FREEZE_LAYERS,
    unfreeze_from=UNFREEZE_FROM
)
model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n🧠 Model Summary:")
print(f"   Total parameters     : {total_params:,}")
print(f"   Trainable parameters : {trainable_params:,}")
print(f"   Frozen parameters    : {total_params - trainable_params:,}")
print(f"   Trainable ratio      : {trainable_params/total_params*100:.1f}%")

# %% [markdown]
# ## 6️⃣ Loss Function, Optimizer & Scheduler

# %%
# ═══════════════════════════════════════════════════════════════
# Compute class weights (handle imbalanced classes)
# ═══════════════════════════════════════════════════════════════
train_labels_array = np.array([s[1] for s in train_samples])
class_freq = train_labels_array.sum(axis=0)
total_samples = len(train_samples)

# Inverse frequency weighting (more weight to rare classes)
pos_weight = (total_samples - class_freq) / (class_freq + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

print("⚖️ Class Weights (pos_weight):")
for i, (name, w) in enumerate(zip(CLASS_NAMES, pos_weight)):
    freq = int(class_freq[i])
    print(f"   {name:25s}: weight={w:.2f}  (freq={freq})")

# ═══════════════════════════════════════════════════════════════
# Loss, Optimizer, Scheduler
# ═══════════════════════════════════════════════════════════════

# BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AdamW optimizer (only trainable parameters)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"\n✅ Training Setup:")
print(f"   Loss      : BCEWithLogitsLoss (weighted)")
print(f"   Optimizer : AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"   Scheduler : CosineAnnealingLR (T_max={NUM_EPOCHS})")

# %% [markdown]
# ## 7️⃣ Training Loop

# %%
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Predictions (threshold at 0.5 on sigmoid outputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader.dataset)
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_f1


print("Training functions defined")

# %%


print(f"{'='*70}")
print(f"🏛️  HERITAGE DEFECT — ResNet50 Training")
print(f"{'='*70}")
print(f"  Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"  Early stopping patience: {PATIENCE} epochs")
print(f"{'='*70}\n")

# Training history
history = {
    'train_loss': [], 'val_loss': [],
    'train_f1': [], 'val_f1': [],
    'lr': []
}

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']
    
    # Train
    train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_loss, val_f1 = validate(model, val_loader, criterion, DEVICE)
    
    # Step scheduler
    scheduler.step()
    
    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    
    # Check for improvement
    improved = ''
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0
        improved = ' ⭐ BEST'
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
            'class_names': CLASS_NAMES,
        }, SAVE_DIR / 'best_resnet50.pth')
    else:
        patience_counter += 1
    
    # Print epoch summary
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} │ "
        f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} │ "
        f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} │ "
        f"LR: {current_lr:.2e}{improved}"
    )
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n{'='*70}")
print(f"✅ Training Complete!")
print(f"   Best Epoch     : {best_epoch}")
print(f"   Best Val F1    : {best_val_f1:.4f}")
print(f"   Model saved to : {SAVE_DIR / 'best_resnet50.pth'}")
print(f"{'='*70}")

# %% [markdown]
# ## Training Visualization

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

epochs_range = range(1, len(history['train_loss']) + 1)

# ── Plot 1: Loss ──
axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', markersize=4, linewidth=2)
axes[0].plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss', markersize=4, linewidth=2)
axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('📉 Loss Curves', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Plot 2: F1 Score ──
axes[1].plot(epochs_range, history['train_f1'], 'b-o', label='Train F1', markersize=4, linewidth=2)
axes[1].plot(epochs_range, history['val_f1'], 'r-s', label='Val F1', markersize=4, linewidth=2)
axes[1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score (Macro)')
axes[1].set_title('📊 F1 Score Curves', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ── Plot 3: Learning Rate ──
axes[2].plot(epochs_range, history['lr'], 'g-^', markersize=4, linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('📈 Learning Rate Schedule', fontweight='bold')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.suptitle('🏛️ ResNet50 Heritage Defect Training History', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9️⃣ Evaluate on Test Set

# %%
# Load best model
checkpoint = torch.load(SAVE_DIR / 'best_resnet50.pth', map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"📥 Loaded best model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")

# Evaluate on test set
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_probs = np.concatenate(all_probs)

# Classification Report
print(f"\n{'='*70}")
print(f"📋 Test Set Classification Report")
print(f"{'='*70}")
print(classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    zero_division=0
))

# Overall metrics
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n📊 Overall Test Metrics:")
print(f"   Macro F1    : {macro_f1:.4f}")
print(f"   Micro F1    : {micro_f1:.4f}")
print(f"   Weighted F1 : {weighted_f1:.4f}")

# %%
# ═══════════════════════════════════════════════════════════════
# Per-Class Confusion Matrices (Multi-label)
# ═══════════════════════════════════════════════════════════════

mcm = multilabel_confusion_matrix(all_labels, all_preds)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i in range(NUM_CLASSES):
    sns.heatmap(
        mcm[i], annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred Neg', 'Pred Pos'],
        yticklabels=['True Neg', 'True Pos'],
        ax=axes[i], cbar=False
    )
    axes[i].set_title(CLASS_NAMES[i], fontsize=11, fontweight='bold')

# Hide the last subplot if odd number of classes
if NUM_CLASSES < len(axes):
    for j in range(NUM_CLASSES, len(axes)):
        axes[j].set_visible(False)

plt.suptitle('🔍 Per-Class Confusion Matrices (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 🔟 Prediction Visualization

# %%
def visualize_predictions(model, dataset, device, n=6):
    """Visualize model predictions on test images."""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    
    for ax_idx, data_idx in enumerate(indices):
        image, true_label = dataset[data_idx]
        
        # Predict
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            probs = torch.sigmoid(output).cpu().numpy()[0]
            preds = (probs > 0.5).astype(int)
        
        # Denormalize image for display
        img = image.numpy().transpose(1, 2, 0)
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        
        axes[ax_idx].imshow(img)
        
        # Format title
        true_classes = [CLASS_NAMES[j] for j in range(NUM_CLASSES) if true_label[j] > 0]
        pred_classes = [f"{CLASS_NAMES[j]} ({probs[j]:.0%})" for j in range(NUM_CLASSES) if preds[j] > 0]
        
        true_str = ', '.join(true_classes) if true_classes else 'None'
        pred_str = ', '.join(pred_classes) if pred_classes else 'None'
        
        # Color code: green if match, red if mismatch
        match = set(true_classes) == set([c.split(' (')[0] for c in pred_classes])
        color = 'green' if match else 'red'
        
        axes[ax_idx].set_title(
            f"True: {true_str}\nPred: {pred_str}",
            fontsize=9, fontweight='bold', color=color
        )
        axes[ax_idx].axis('off')
    
    plt.suptitle('🔮 Model Predictions on Test Images (Green=Correct, Red=Incorrect)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_predictions(model, test_dataset, DEVICE, n=6)

# %%
# ═══════════════════════════════════════════════════════════════
# Per-Class Confidence Distribution
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i in range(NUM_CLASSES):
    pos_mask = all_labels[:, i] == 1
    neg_mask = all_labels[:, i] == 0
    
    if pos_mask.sum() > 0:
        axes[i].hist(all_probs[pos_mask, i], bins=20, alpha=0.7, color='green', label='Positive', density=True)
    if neg_mask.sum() > 0:
        axes[i].hist(all_probs[neg_mask, i], bins=20, alpha=0.7, color='red', label='Negative', density=True)
    
    axes[i].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    axes[i].set_title(CLASS_NAMES[i], fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=7)
    axes[i].set_xlabel('Confidence')

if NUM_CLASSES < len(axes):
    for j in range(NUM_CLASSES, len(axes)):
        axes[j].set_visible(False)

plt.suptitle('📊 Per-Class Confidence Distributions (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 1️⃣1️⃣ Export & Summary

# %%
# ═══════════════════════════════════════════════════════════════
# Save final model for inference
# ═══════════════════════════════════════════════════════════════

# Save just the model weights (lighter file)
torch.save(model.state_dict(), SAVE_DIR / 'resnet50_weights_only.pth')

# Save training history
history_df = pd.DataFrame(history)
history_df.index = range(1, len(history_df) + 1)
history_df.index.name = 'epoch'
history_df.to_csv(SAVE_DIR / 'training_history.csv')

print("📁 Saved Files:")
print(f"   ├── {SAVE_DIR / 'best_resnet50.pth'}          (full checkpoint)")
print(f"   ├── {SAVE_DIR / 'resnet50_weights_only.pth'}  (weights only)")
print(f"   ├── {SAVE_DIR / 'training_history.csv'}       (loss & metrics)")
print(f"   ├── {SAVE_DIR / 'training_history.png'}       (plots)")
print(f"   ├── {SAVE_DIR / 'confusion_matrices.png'}     (per-class CM)")
print(f"   ├── {SAVE_DIR / 'predictions.png'}            (sample predictions)")
print(f"   └── {SAVE_DIR / 'confidence_distributions.png'}")

print(f"\n{'='*70}")
print(f"🏛️  HERITAGE DEFECT — ResNet50 Training Summary")
print(f"{'='*70}")
print(f"  Model         : ResNet50 (ImageNet V2 pre-trained)")
print(f"  Task          : Multi-label classification ({NUM_CLASSES} classes)")
print(f"  Train images  : {len(train_samples)}")
print(f"  Best epoch    : {best_epoch}")
print(f"  Best Val F1   : {best_val_f1:.4f}")
print(f"  Test Macro F1 : {macro_f1:.4f}")
print(f"  Test Micro F1 : {micro_f1:.4f}")
print(f"{'='*70}")

# %% [markdown]
# ## 1️⃣2️⃣ Quick Inference Function
# 
# Use this cell to predict defects on any new image:

# %%
def predict_image(image_path, model, device, threshold=0.5):
    """
    Predict defects in a heritage building image.
    
    Args:
        image_path: Path to the image file
        model: Trained ResNet50 model
        device: torch device
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: {class_name: confidence_score}
    """
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    transformed = val_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(transformed)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    # Get detected defects
    results = {}
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        if prob >= threshold:
            results[name] = float(prob)
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    
    ax1.imshow(image)
    ax1.set_title('Input Image', fontweight='bold')
    ax1.axis('off')
    
    colors = ['#2ecc71' if p >= threshold else '#e74c3c' for p in probs]
    bars = ax2.barh(CLASS_NAMES, probs, color=colors, edgecolor='white')
    ax2.axvline(x=threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Defect Predictions', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results


# ── Example: predict on a test image ──
test_image_path = test_samples[0][0]
print(f"🔮 Predicting: {Path(test_image_path).name}\n")
results = predict_image(test_image_path, model, DEVICE)

print(f"\n🏷️ Detected Defects:")
for defect, conf in sorted(results.items(), key=lambda x: -x[1]):
    print(f"   • {defect}: {conf:.1%}")

# %%




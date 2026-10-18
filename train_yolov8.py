"""
HERITAGEDEFECT - YOLOv8 Segmentation Training Script
=====================================================
Trains a YOLOv8 nano segmentation model on the Heritage Defect dataset.
11 classes: Blackening, Brick Spalling, Concrete Spalling, Corrosion,
Crack, Dampness, Dampness with fungus, Efflorescence, Flaking Plaster,
Moss, Vegetation

Dataset: 711 train / 59 valid / 29 test images
Format: YOLOv8 polygon segmentation
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    # ── Configuration ──────────────────────────────────────────────
    PROJECT_DIR = Path(__file__).parent
    DATA_YAML = PROJECT_DIR / "data.yaml"
    
    # Training hyperparameters
    MODEL_SIZE = "yolov8n-seg"  # nano model (fast training, good baseline)
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH_SIZE = 8              # conservative for Mac (CPU/MPS)
    PATIENCE = 15               # early stopping patience
    
    # Output directory
    PROJECT_NAME = str(PROJECT_DIR / "runs")
    EXPERIMENT_NAME = "heritage_defect_seg"
    
    # ── Device Selection ───────────────────────────────────────────
    import torch
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("🍎 Using Apple MPS (Metal Performance Shaders) for training")
    elif torch.cuda.is_available():
        device = "0"    # NVIDIA GPU
        print("🎮 Using CUDA GPU for training")
    else:
        device = "cpu"
        print("💻 Using CPU for training (this will be slower)")
    
    # ── Validate Dataset ───────────────────────────────────────────
    if not DATA_YAML.exists():
        print(f"❌ data.yaml not found at {DATA_YAML}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🏛️  HERITAGEDEFECT - YOLOv8 Segmentation Training")
    print(f"{'='*60}")
    print(f"📁 Dataset config : {DATA_YAML}")
    print(f"🤖 Model          : {MODEL_SIZE}")
    print(f"📐 Image size     : {IMG_SIZE}x{IMG_SIZE}")
    print(f"📦 Batch size     : {BATCH_SIZE}")
    print(f"🔄 Epochs         : {EPOCHS}")
    print(f"⏱️  Early stopping : {PATIENCE} epochs patience")
    print(f"💻 Device         : {device}")
    print(f"{'='*60}\n")
    
    # ── Load Model ─────────────────────────────────────────────────
    print("📥 Loading YOLOv8 nano segmentation model...")
    model = YOLO(MODEL_SIZE)
    
    # ── Train ──────────────────────────────────────────────────────
    print("🚀 Starting training...\n")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        
        # Optimization
        optimizer="auto",          # AdamW
        lr0=0.01,                  # initial learning rate
        lrf=0.01,                  # final learning rate factor
        warmup_epochs=3.0,         # warmup epochs
        
        # Augmentation
        hsv_h=0.015,              # hue augmentation
        hsv_s=0.7,                # saturation augmentation
        hsv_v=0.4,                # value augmentation
        degrees=10.0,             # rotation augmentation
        translate=0.1,            # translation augmentation
        scale=0.5,                # scale augmentation
        fliplr=0.5,              # horizontal flip probability
        flipud=0.5,              # vertical flip probability (heritage images can be any orientation)
        mosaic=1.0,              # mosaic augmentation
        
        # Logging
        plots=True,               # generate training plots
        save=True,                # save checkpoints
        verbose=True,
    )
    
    # ── Validation ─────────────────────────────────────────────────
    print("\n📊 Running validation on best model...")
    best_model_path = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
    
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        val_results = best_model.val(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            device=device,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"📁 Results saved to: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
        print(f"🏆 Best model: {best_model_path}")
        print(f"{'='*60}")
    else:
        print(f"⚠️  Best model not found at {best_model_path}")
        print("Training may have been interrupted.")

if __name__ == "__main__":
    main()

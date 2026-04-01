"""
YOLOv8-nano Gate Detector Trainer
==================================
Trains YOLOv8-nano on the images collected by collect_training_data.py.
On RTX 4080: ~5-10 minutes for 50 epochs.

Run:
    python train_gate_detector.py

Output:
    gate_detector.pt  — copy of the best weights, ready for V16
"""

import shutil
import yaml
from pathlib import Path


TRAINING_DATA = Path("training_data")
DATASET_YAML  = Path("dataset.yaml")
EPOCHS        = 50
IMG_SIZE      = 640
BATCH         = 16        # 4080 can handle 16 easily; increase to 32 if you have headroom
DEVICE        = 0         # GPU 0 (RTX 4080)
OUTPUT_NAME   = "gate_detector"


def check_data():
    """Verify training data exists and print stats."""
    images = list((TRAINING_DATA / "images").glob("*.jpg"))
    labels = list((TRAINING_DATA / "labels").glob("*.txt"))

    print(f"Training data: {len(images)} images, {len(labels)} labels")
    if len(images) == 0:
        print("ERROR: No images found. Run collect_training_data.py first.")
        return False
    if len(images) < 200:
        print(f"WARNING: Only {len(images)} images. Recommend 500+ for good results.")
        print("Consider running collect_training_data.py longer.")
    return True


def create_dataset_yaml():
    """Generate YOLO dataset.yaml with 80/20 train/val split."""
    images = sorted((TRAINING_DATA / "images").glob("*.jpg"))
    n = len(images)
    split = int(n * 0.8)

    train_imgs = images[:split]
    val_imgs   = images[split:]

    # Write train/val image lists
    (TRAINING_DATA / "train.txt").write_text(
        "\n".join(str(p.resolve()) for p in train_imgs)
    )
    (TRAINING_DATA / "val.txt").write_text(
        "\n".join(str(p.resolve()) for p in val_imgs)
    )

    dataset = {
        "path": str(TRAINING_DATA.resolve()),
        "train": "train.txt",
        "val":   "val.txt",
        "nc":    1,
        "names": ["gate"],
    }

    with open(DATASET_YAML, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False)

    print(f"Dataset split: {len(train_imgs)} train, {len(val_imgs)} val")
    return str(DATASET_YAML.resolve())


def train():
    from ultralytics import YOLO
    import torch

    print("=" * 60)
    print("YOLOv8-nano Gate Detector Training")
    print("=" * 60)

    cuda = torch.cuda.is_available()
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No CUDA GPU found. Training on CPU will be slow.")

    if not check_data():
        return

    yaml_path = create_dataset_yaml()
    print(f"Dataset YAML: {yaml_path}")
    print(f"Epochs: {EPOCHS}, Image size: {IMG_SIZE}, Batch: {BATCH}")
    print(f"Device: {'GPU 0' if cuda else 'CPU'}\n")

    # Load YOLOv8-nano base model (downloads pretrained weights ~6MB)
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE if cuda else "cpu",
        name=OUTPUT_NAME,
        patience=15,          # Stop early if no improvement for 15 epochs
        save=True,
        plots=True,
        verbose=True,
        # Augmentation (built-in YOLOv8)
        hsv_h=0.015,          # Hue variation (gates are orange, keep narrow)
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10.0,         # Random rotation
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.5,
    )

    # Copy best weights to project root
    best_weights = Path(f"runs/detect/{OUTPUT_NAME}/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, "gate_detector.pt")
        print(f"\nBest weights copied to gate_detector.pt")
    else:
        print(f"\nWARNING: Could not find {best_weights}")
        print("Check runs/detect/ for training outputs.")
        return

    # Print final metrics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Validate on val set
    print("\nValidating on held-out images...")
    val_results = model.val(data=yaml_path, device=DEVICE if cuda else "cpu")

    map50 = val_results.box.map50
    map50_95 = val_results.box.map

    print(f"\nmAP50:    {map50:.3f}   (target: ≥0.80)")
    print(f"mAP50-95: {map50_95:.3f}")

    if map50 >= 0.80:
        print("\n[PASS] mAP50 >= 0.80 — detector is good to use!")
    elif map50 >= 0.60:
        print("\n[OK] mAP50 is decent. Consider collecting more training data.")
    else:
        print("\n[WARN] mAP50 is low. Try collecting more varied training data.")

    print(f"\nReady to race! Run:")
    print(f"  python airsim_contour_trackerV16.py")


if __name__ == "__main__":
    train()

"""
YOLO Training Script for Member B (Detection Lead).
Lead: Member B (Shaurya Vaid) • Branch: feature/detection

Trains Ultralytics YOLO (v8 or v11) on Traffic Sign Detection datasets (GTSDB or custom YOLO format).
Once training completes, automatically copies best.pt to weights/detection/best.pt.
"""

import os
import shutil
import argparse
from pathlib import Path


def generate_template_yaml(yaml_path: Path):
    """Generates a starter YAML config for YOLO dataset formatting."""
    content = """# YOLO Traffic Sign Dataset Configuration
path: ../data/detection_dataset # dataset root dir
train: images/train
val: images/val

# Classes
names:
  0: traffic_sign
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[Info] Created starter dataset config at: {yaml_path}")


def train_yolo(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu"
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Ultralytics is not installed. Run: pip install ultralytics")
        return

    yaml_file = Path(data_yaml)
    if not yaml_file.exists():
        print(f"[Warning] Dataset yaml '{data_yaml}' not found. Generating starter template...")
        generate_template_yaml(yaml_file)

    print(f"\n[Training] Starting YOLO training using base model '{base_model}' for {epochs} epochs...")
    model = YOLO(base_model)
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="traffic_sign_detector"
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[Success] Training complete! Best weights saved to: {best_pt}")
        
        # Deploy to module B weights
        dest_mod_b = Path(__file__).resolve().parent / "weights" / "best.pt"
        dest_mod_b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest_mod_b)
        print(f"[Deployed] Copied to: {dest_mod_b}")

        # Deploy to integration pipeline weights
        dest_root = Path(__file__).resolve().parent.parent.parent / "weights" / "detection" / "best.pt"
        dest_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest_root)
        print(f"[Deployed] Copied to master integration pipeline: {dest_root}")
    else:
        print("[Warning] Could not locate output best.pt file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member B YOLO Trainer")
    parser.add_argument("--data", type=str, default="data/detection_data.yaml", help="Path to data.yaml")
    parser.add_argument("--base", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda:0, etc.)")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

import sys
sys.path.insert(0, '.')
import os
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.train_custom import get_model_definition
from src.gtsrb_classes import GTSRB_CLASSES

ARCHIVE_PATH = r"C:\Users\Sombuddha Ash\Downloads\archive.zip"

def augment_clean_icon(img, count=60):
    """
    Augments a clean Meta icon with various realistic transforms:
    - Random scaling and padding (simulates distance)
    - Random background (white, gray, outdoor noise, blurred texture)
    - Random rotation (-12 to +12 degrees)
    - Random contrast and brightness
    - Random Gaussian blur / noise
    """
    augmented = []
    h, w = img.shape[:2]
    
    # If transparent (RGBA), composite on white
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        bgr = img[:, :, :3]
        white_bg = np.ones_like(bgr) * 255
        img = (bgr * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
    
    for _ in range(count):
        # 1. Random rotation
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.75, 1.05)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        # Background color: mix of white, light gray, outdoor tones
        bg_val = int(np.random.choice([255, 240, 220, 180, 120, 60]))
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(bg_val, bg_val, bg_val))
        
        # 2. Random brightness/contrast
        alpha_bc = np.random.uniform(0.7, 1.3)
        beta_bc = np.random.uniform(-30, 30)
        adj = cv2.convertScaleAbs(rotated, alpha=alpha_bc, beta=beta_bc)
        
        # 3. Random blur
        if np.random.random() > 0.5:
            k = np.random.choice([3, 5])
            adj = cv2.GaussianBlur(adj, (k, k), 0)
            
        # 4. Resize to 32x32
        res = cv2.resize(adj, (32, 32), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        chw = np.transpose(norm, (2, 0, 1))
        augmented.append(chw)
        
    return augmented

def load_data(samples_per_train=120, meta_aug_count=80):
    images = []
    labels = []
    
    print("Loading data from archive.zip...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        # 1. Load and augment Meta icons for all 43 classes
        for cid in range(43):
            meta_name = f"Meta/{cid}.png"
            if meta_name in z.namelist():
                with z.open(meta_name) as f:
                    b = np.frombuffer(f.read(), np.uint8)
                    icon = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
                    if icon is not None:
                        augs = augment_clean_icon(icon, count=meta_aug_count)
                        images.extend(augs)
                        labels.extend([cid] * len(augs))
                        
        print(f"Loaded {len(images)} augmented Meta samples across 43 classes.")
        
        # 2. Load Train samples from archive
        class_files = {i: [] for i in range(43)}
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[0].lower() == "train" and parts[1].isdigit():
                cid = int(parts[1])
                if 0 <= cid < 43 and name.lower().endswith((".png", ".ppm", ".jpg")):
                    if len(class_files[cid]) < samples_per_train:
                        class_files[cid].append(name)
                        
        train_count = 0
        for cid in range(43):
            for fname in class_files[cid]:
                with z.open(fname) as f:
                    b = np.frombuffer(f.read(), np.uint8)
                    raw_img = cv2.imdecode(b, cv2.IMREAD_COLOR)
                    if raw_img is not None:
                        res = cv2.resize(raw_img, (32, 32), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        norm = rgb.astype(np.float32) / 255.0
                        chw = np.transpose(norm, (2, 0, 1))
                        images.append(chw)
                        labels.append(cid)
                        train_count += 1
                        
        print(f"Loaded {train_count} real Train samples from archive. Total samples: {len(images)}")
        
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

if __name__ == "__main__":
    X, y = load_data(samples_per_train=100, meta_aug_count=80)
    print(f"Total dataset shape: X={X.shape}, y={y.shape}")

    # Shuffle
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    device = torch.device("cpu")
    model = get_model_definition(43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print("Training model for 12 epochs...")
    for epoch in range(12):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)
            pred = out.argmax(dim=1)
            correct += (pred == by).sum().item()
            total += len(by)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vo = model(vx)
                val_correct += (vo.argmax(dim=1) == vy).sum().item()
                val_total += len(vy)

        print(f"Epoch {epoch+1:2d}/12 | Train Loss: {running_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | Val Acc: {val_correct/val_total*100:.2f}%")

    # Now evaluate on user's crop!
    model.eval()
    user_crop = cv2.imread('scratch/user_sign_crop.png')
    res = cv2.resize(user_crop, (32, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0
    inp = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        top3 = np.argsort(probs)[::-1][:3]
        print("\n--- EVALUATION ON USER UPLOADED IMAGE ---")
        for idx in top3:
            print(f"  Class {idx:2d} ({GTSRB_CLASSES[idx]}): {probs[idx]*100:.2f}%")

    # Evaluate on all 43 clean Meta icons
    meta_correct = 0
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        for cid in range(43):
            meta_name = f"Meta/{cid}.png"
            if meta_name in z.namelist():
                with z.open(meta_name) as f:
                    b = np.frombuffer(f.read(), np.uint8)
                    icon = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
                    if icon is not None:
                        if icon.shape[2] == 4:
                            a = icon[:, :, 3] / 255.0
                            white = np.ones_like(icon[:, :, :3]) * 255
                            icon = (icon[:, :, :3] * a[:, :, None] + white * (1 - a[:, :, None])).astype(np.uint8)
                        r = cv2.resize(icon, (32, 32))
                        rg = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                        t = torch.from_numpy(np.transpose(rg.astype(np.float32)/255.0, (2, 0, 1))).unsqueeze(0)
                        with torch.no_grad():
                            pred_cid = model(t).argmax(dim=1).item()
                            if pred_cid == cid:
                                meta_correct += 1
    print(f"\n--- EVALUATION ON CLEAN META ICONS ---")
    print(f"Meta Accuracy: {meta_correct}/43 ({meta_correct/43*100:.2f}%)")

    # Save candidate weights
    torch.jit.trace(model, inp).save("scratch/test_classifier.pt")
    print("Saved candidate model to scratch/test_classifier.pt")

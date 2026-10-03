"""
Custom Traffic Sign CNN Trainer and Exporter.
Supports training on custom datasets or full GTSRB training split.
Exports optimized PyTorch/TorchScript weights for production deployment.
"""

from typing import Optional, Dict, Tuple
import os
import time
import json
import numpy as np
import cv2


def get_model_definition(num_classes: int = 43):
    import torch
    import torch.nn as nn

    class TrafficSignCNN(nn.Module):
        def __init__(self, num_classes: int = 43):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 16x16
                nn.Dropout2d(0.2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 8x8
                nn.Dropout2d(0.25),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 4x4
                nn.Dropout2d(0.3)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    return TrafficSignCNN(num_classes=num_classes)


def train_model(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    num_classes: int = 43,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    output_path: str = "weights/classification/classifier.pt",
    progress_callback = None
) -> Tuple[float, str]:
    """
    Trains TrafficSignCNN on in-memory numpy arrays and saves TorchScript model.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare datasets
    tensor_x = torch.from_numpy(train_images).float()
    tensor_y = torch.from_numpy(train_labels).long()
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_images is not None and val_labels is not None:
        val_x = torch.from_numpy(val_images).float()
        val_y = torch.from_numpy(val_labels).long()
        val_dataset = TensorDataset(val_x, val_y)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model_definition(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = correct / total if total > 0 else 0.0

        val_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_out = model(vx)
                    _, v_pred = torch.max(v_out.data, 1)
                    val_total += vy.size(0)
                    val_correct += (v_pred == vy).sum().item()
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}%")

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_acc, val_acc)

    # Save as TorchScript for universal deployment
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.eval()
    example_input = torch.rand(1, 3, 32, 32).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_path)
    print(f"[Success] Exported TorchScript model to: {output_path}")

    return train_acc, output_path

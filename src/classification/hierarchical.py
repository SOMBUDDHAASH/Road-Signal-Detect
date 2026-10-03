"""
Hierarchical Classification Architecture & Two-Level Loss Function.
Maintained by Member D (Integration & Pipeline Lead).

Implements:
1. Two-level hierarchy: Super-Category (Prohibitory, Danger, Mandatory, Other) + Fine-Grained GTSRB Class (43 classes).
2. Joint Multi-Task Hierarchical Loss: L_total = L_fine + lambda_cat * L_super.
3. Universal deployment wrapper with TorchScript export compatibility.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.schema import SignCategory
from src.gtsrb_classes import GTSRB_CATEGORIES

# Map SignCategory Enum to Integer ID [0..3]
CATEGORY_TO_ID: Dict[SignCategory, int] = {
    SignCategory.PROHIBITORY: 0,
    SignCategory.DANGER: 1,
    SignCategory.MANDATORY: 2,
    SignCategory.OTHER: 3,
}

# Precompute mapping from ClassId (0..42) to SuperCategory ID (0..3)
CLASS_TO_CATEGORY_ID: Dict[int, int] = {
    cid: CATEGORY_TO_ID[GTSRB_CATEGORIES.get(cid, SignCategory.OTHER)]
    for cid in range(43)
}


class HierarchicalTrafficSignCNN(nn.Module):
    """
    Deep CNN with shared representation trunk and dual classification heads:
    - Head 1: SuperCategory (4 classes)
    - Head 2: FineGrained Sign Class (43 classes)
    """

    def __init__(self, num_classes: int = 43, num_categories: int = 4, export_fine_only: bool = True):
        super().__init__()
        self.export_fine_only = export_fine_only

        # Feature Extractor Trunk
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(0.20),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x4
            nn.Dropout2d(0.30),
        )

        # Shared Dense Representation
        self.fc_shared = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
        )

        # Dual Heads
        self.head_category = nn.Linear(256, num_categories)
        self.head_fine = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        shared = self.fc_shared(flat)

        fine_logits = self.head_fine(shared)

        if self.export_fine_only and not self.training:
            return fine_logits

        cat_logits = self.head_category(shared)
        return fine_logits, cat_logits


class HierarchicalLoss(nn.Module):
    """
    Two-level loss function combining Cross-Entropy on fine-grained classes
    with Cross-Entropy on super-categories.
    """

    def __init__(self, lambda_category: float = 0.50):
        super().__init__()
        self.lambda_category = lambda_category
        self.criterion_fine = nn.CrossEntropyLoss()
        self.criterion_cat = nn.CrossEntropyLoss()

    def forward(
        self,
        fine_logits: torch.Tensor,
        cat_logits: torch.Tensor,
        targets_fine: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Derive target category IDs from target fine-grained class IDs
        device = targets_fine.device
        cat_map = torch.tensor([CLASS_TO_CATEGORY_ID[i] for i in range(43)], device=device)
        targets_cat = cat_map[targets_fine]

        loss_fine = self.criterion_fine(fine_logits, targets_fine)
        loss_cat = self.criterion_cat(cat_logits, targets_cat)

        loss_total = loss_fine + self.lambda_category * loss_cat
        return loss_total, loss_fine, loss_cat

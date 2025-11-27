import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

from sod_model import SaliencyNet
from data_loader import SaliencyDataset

SEED = 31
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


images_dir = "data/DUTS/resized_images_224"
masks_dir  = "data/DUTS/resized_masks_224"

full_dataset = SaliencyDataset(images_dir, masks_dir, transform=None)
N = len(full_dataset)
print("Total dataset size:", N)


indexes = list(range(N))
np.random.seed(SEED)
np.random.shuffle(indexes)

train_end = int(0.7 * N)
val_end   = int(0.85 * N)
test_ids  = indexes[val_end:]

test_dataset = Subset(full_dataset, test_ids)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Test set size:", len(test_dataset))


model = SaliencyNet().to(device)
model.load_state_dict(torch.load("best_sod_model.pth", map_location=device))
model.eval()
print("Loaded best_sod_model.pth")

criterion = nn.BCELoss()
eps = 1e-6

def iou_loss(preds, targets, eps=1e-6):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()


test_loss = 0.0
total_tp = total_fp = total_fn = 0.0
total_abs_err = 0.0
total_pixels = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks  = masks.to(device).float()


        preds = model(images)


        bce  = criterion(preds, masks)
        iouL = iou_loss(preds, masks)
        loss = bce + iouL
        test_loss += loss.item()


        bin_preds = (preds > 0.5).float()


        bin_preds_f = bin_preds.view(-1)
        masks_f     = masks.view(-1)

        tp = (bin_preds_f * masks_f).sum()
        fp = (bin_preds_f * (1 - masks_f)).sum()
        fn = ((1 - bin_preds_f) * masks_f).sum()

        total_tp += tp.item()
        total_fp += fp.item()
        total_fn += fn.item()


        total_abs_err += torch.abs(preds - masks).sum().item()
        total_pixels  += masks.numel()


test_loss = test_loss / len(test_loader)
precision = total_tp / (total_tp + total_fp + eps)
recall    = total_tp / (total_tp + total_fn + eps)
f1        = 2 * precision * recall / (precision + recall + eps)
iou       = total_tp / (total_tp + total_fp + total_fn + eps)
mae       = total_abs_err / (total_pixels + eps)


print("\n===== FINAL TEST RESULTS =====")
print(f"Test Loss : {test_loss:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"MAE       : {mae:.4f}")

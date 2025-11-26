import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sod_model import SaliencyNet
from data_loader import SaliencyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_dataset = SaliencyDataset(
    "data/DUTS/resized_images_224",
    "data/DUTS/resized_masks_224"
)
print("Test dataset size:", len(test_dataset))

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

model = SaliencyNet().to(device)
criterion = nn.BCELoss()

model.load_state_dict(torch.load(
    "best_sod_model.pth",
    map_location=device
))
print("Loaded best_sod_model.pth")

print("\nEvaluating on test set...")
model.eval()

test_loss = 0.0
eps = 1e-6
total_tp = total_fp = total_fn = total_tn = 0.0
total_abs_err = 0.0
total_pixels = 0

def iou_loss(preds, targets, eps=1e-6):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    iou = (intersection + eps) / (union - intersection + eps)
    return 1 - iou.mean()

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks  = masks.to(device).float()

        logits = model(images)
        preds  = torch.sigmoid(logits)

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
        tn = ((1 - bin_preds_f) * (1 - masks_f)).sum()

        total_tp += tp.item()
        total_fp += fp.item()
        total_fn += fn.item()
        total_tn += tn.item()

        total_abs_err += torch.abs(preds - masks).sum().item()
        total_pixels  += masks.numel()

test_loss = test_loss / len(test_loader)
precision = total_tp / (total_tp + total_fp + eps)
recall    = total_tp / (total_tp + total_fn + eps)
f1        = 2 * precision * recall / (precision + recall + eps)
iou       = total_tp / (total_tp + total_fp + total_fn + eps)
mae       = total_abs_err / (total_pixels + eps)

print("\n===== TEST RESULTS =====")
print(f"Test Loss : {test_loss:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"MAE       : {mae:.4f}")

images, masks = next(iter(test_loader))
images = images.to(device)
masks  = masks.to(device)

with torch.no_grad():
    logits = model(images)
    preds  = torch.sigmoid(logits)

images_cpu = images.cpu()
masks_cpu  = masks.cpu()
preds_cpu  = preds.cpu()

bin_preds = (preds_cpu > 0.5).float()

num_show = min(3, images_cpu.size(0))
plt.figure(figsize=(16, 4 * num_show))

for i in range(num_show):
    img  = images_cpu[i].permute(1, 2, 0)
    gt   = masks_cpu[i].squeeze()
    pred = bin_preds[i].squeeze()

    plt.subplot(num_show, 4, 4*i + 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(num_show, 4, 4*i + 2)
    plt.title("GT Mask")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    plt.subplot(num_show, 4, 4*i + 3)
    plt.title("Pred Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(num_show, 4, 4*i + 4)
    plt.title("Overlay")
    plt.imshow(img)
    plt.imshow(pred, cmap="jet", alpha=0.4)
    plt.axis("off")

plt.tight_layout()
plt.show()

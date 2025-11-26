import os
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from data_loader import SaliencyDataset
from sod_model import SaliencyNet


SEED = 31


def iou_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection + eps) / (union + eps)


def precision_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)
    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)
    tp = (pred * mask).sum()
    fn = ((1 - pred) * mask).sum()
    return (tp + eps) / (tp + fn + eps)


def f1_score(pred, mask, eps=1e-6):
    p = precision_score(pred, mask, eps)
    r = recall_score(pred, mask, eps)
    return 2 * (p * r) / (p + r + eps)


def mae_score(pred, mask):
    return torch.abs(pred - mask).mean()


def build_test_loader(batch_size=64):
    images_dir = "data/DUTS/resized_images_224"
    masks_dir = "data/DUTS/resized_masks_224"

    base_dataset = SaliencyDataset(images_dir, masks_dir, transform=None)
    N = len(base_dataset)

    indexes = list(range(N))
    random.seed(SEED)
    random.shuffle(indexes)

    train_end = int(0.7 * N)
    val_end = int(0.85 * N)
    test_ids = indexes[val_end:]

    eval_dataset_full = SaliencyDataset(images_dir, masks_dir, transform=None)
    test_dataset = Subset(eval_dataset_full, test_ids)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def load_model(device):
    model = SaliencyNet().to(device)
    checkpoint_path = "best_sod_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading test set...")
    test_loader = build_test_loader(batch_size=64)

    print("Loading model...")
    model = load_model(device)

    criterion = nn.BCELoss()
    eps = 1e-6

    test_loss = 0.0
    iou_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    mae_list = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            preds = model(images)  # sigmoid output in [0,1]

            bce = criterion(preds, masks)

            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)
            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection
            iou_batch = (intersection + eps) / (union + eps)
            iou_loss = 0.5 * (1 - iou_batch.mean())

            loss = bce + iou_loss
            test_loss += loss.item()

            bin_preds = (preds > 0.5).float()

            iou_list.append(iou_score(bin_preds, masks).item())
            prec_list.append(precision_score(bin_preds, masks).item())
            rec_list.append(recall_score(bin_preds, masks).item())
            f1_list.append(f1_score(bin_preds, masks).item())
            mae_list.append(mae_score(bin_preds, masks).item())

    test_loss = test_loss / len(test_loader)
    iou = float(np.mean(iou_list))
    prec = float(np.mean(prec_list))
    rec = float(np.mean(rec_list))
    f1 = float(np.mean(f1_list))
    mae = float(np.mean(mae_list))

    print("\n------------------ MODEL PERFORMANCE ------------------")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"IoU           : {iou:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"MAE           : {mae:.4f}")
    print("--------------------------------------------------------")

    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        preds = model(images)

    images_cpu = images.cpu()
    masks_cpu = masks.cpu()
    preds_cpu = preds.cpu()
    bin_preds = (preds_cpu > 0.5).float()

    num_show = min(3, images_cpu.size(0))
    plt.figure(figsize=(16, 4 * num_show))

    for i in range(num_show):
        img = images_cpu[i].permute(1, 2, 0)
        gt = masks_cpu[i].squeeze()
        pred = bin_preds[i].squeeze()

        plt.subplot(num_show, 4, 4 * i + 1)
        plt.title("Input image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(num_show, 4, 4 * i + 2)
        plt.title("GT saliency mask")
        plt.imshow(gt, cmap="gray")
        plt.axis("off")

        plt.subplot(num_show, 4, 4 * i + 3)
        plt.title("Predicted mask")
        plt.imshow(pred, cmap="gray")
        plt.axis("off")

        plt.subplot(num_show, 4, 4 * i + 4)
        plt.title("Overlay (pred + input)")
        plt.imshow(img)
        plt.imshow(pred, cmap="jet", alpha=0.4)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()

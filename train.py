import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from data_loader import SaliencyDataset, Augment
from sod_model import SaliencyNet


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
SEED = 31
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------
# DICE LOSS
# -----------------------------
def dice_loss(preds, targets, eps=1e-6):
    # preds, targets: (B, 1, H, W)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


# -----------------------------
# LOAD DATASETS (WITH/WITHOUT AUG)
# -----------------------------
images_dir = "data/ECSSD/resized_images_128"
masks_dir  = "data/ECSSD/resized_masks_128"

# full dataset used only to get length / indices
base_dataset = SaliencyDataset(images_dir, masks_dir, transform=None)
N = len(base_dataset)

indexes = list(range(N))
random.seed(SEED)
random.shuffle(indexes)

train_end = int(0.70 * N)
val_end   = int(0.85 * N)

train_ids = indexes[:train_end]
val_ids   = indexes[train_end:val_end]
test_ids  = indexes[val_end:]

# train dataset WITH augmentation
train_dataset_full = SaliencyDataset(images_dir, masks_dir, transform=Augment())
# val/test datasets WITHOUT augmentation
val_dataset_full   = SaliencyDataset(images_dir, masks_dir, transform=None)
test_dataset_full  = SaliencyDataset(images_dir, masks_dir, transform=None)

train_dataset = Subset(train_dataset_full, train_ids)
val_dataset   = Subset(val_dataset_full,   val_ids)
test_dataset  = Subset(test_dataset_full,  test_ids)


# -----------------------------
# DATALOADERS
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)


# -----------------------------
# MODEL + OPTIMIZER
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = SaliencyNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 120
best_val_loss = float('inf')


# -----------------------------
# TRAINING + VALIDATION
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(images)

        bce = criterion(preds, masks)
        dice = dice_loss(preds, masks)
        loss = bce + dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            bce = criterion(preds, masks)
            dice = dice_loss(preds, masks)
            loss = bce + dice

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f"           Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_sod_model.pth')
        print("           ✔ Saved best model")


# -----------------------------
# LOAD BEST MODEL FOR TESTING
# -----------------------------
model.load_state_dict(torch.load('best_sod_model.pth', map_location=device))


# -----------------------------
# TEST EVALUATION
# -----------------------------
print("\nEvaluating on test set...")
model.eval()
test_loss = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        bce = criterion(preds, masks)
        dice = dice_loss(preds, masks)
        loss = bce + dice

        test_loss += loss.item()

test_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_loss:.4f}")


# -----------------------------
# VISUALIZATION (ON BEST MODEL)
# -----------------------------
images, masks = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    preds = model(images).cpu()

bin_preds = (preds > 0.5).float()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(images[0].cpu().permute(1, 2, 0))
plt.subplot(1, 3, 2); plt.title("GT Mask"); plt.imshow(masks[0].squeeze(), cmap="gray")
plt.subplot(1, 3, 3); plt.title("Pred Mask"); plt.imshow(bin_preds[0].squeeze(), cmap="gray")
plt.show()

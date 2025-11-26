import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Subset

from data_loader import SaliencyDataset, Augment
from sod_model import SaliencyNet

#reproducibility (We use the same seed here so if we run again, we get the same splits, same initial weights)
SEED = 31
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


#Loss function: IoU loss
"""IoU is how much the predicted mask overlaps with the true
    mask, divided by how big their combined area is"""

def iou_loss(preds, targets, eps=1e-6):
    """preds, targets: tensors with shape (B, 1, H, W), values in [0, 1]
        We compute a differentiable IoU over the batch, then return
        loss = 0.5 * (1 - IoU), higher the IoU, smaller the loss """


    #Flatten each image to a vector: (B, H*W) #B - batch size, H*W the number of pixels for image
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    #Intersection and union for each item in the batch
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    #IoU per item, then mean over batch
    iou = (intersection + eps) / (union + eps)
    iou_mean = iou.mean()

    return 0.5 * (1 - iou_mean)



#Load datasets
images_dir = 'data/DUTS/resized_images_224'
masks_dir = 'data/DUTS/resized_masks_224'

#full dataset used only to get length / indices
base_dataset = SaliencyDataset(images_dir, masks_dir, transform=None)
N = len(base_dataset)

#create shuffled indices
indexes = list(range(N))
random.seed(SEED)
random.shuffle(indexes)

#spliting the dataset 70/15/15
train_end = int(0.7 * N)
val_end = int(0.85 * N)

train_ids = indexes[:train_end]
val_ids = indexes[train_end:val_end]
test_ids = indexes[val_end:]

#train dataset with augmentation
train_dataset_full = SaliencyDataset(images_dir, masks_dir, transform=Augment())

eval_dataset_full = SaliencyDataset(images_dir, masks_dir, transform=None)

train_dataset = Subset(train_dataset_full, train_ids)
val_dataset = Subset(eval_dataset_full, val_ids)
test_dataset = Subset(eval_dataset_full, test_ids)


#Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


#Model + Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = SaliencyNet().to(device)

#predictions are sigmoid outputs, so we use Binary Cross-Entropy as the training loss
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5,)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=4,
)

num_epochs = 40
best_val_loss = float('inf')

#Train and Validation

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()

        preds = model(images)       #preds in [0, 1] because of sigmoid

        bce = criterion(preds, masks)
        iouL = iou_loss(preds, masks)
        loss = bce + iouL

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

    #validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            preds = model(images)
            bce = criterion(preds, masks)
            iouL = iou_loss(preds, masks)

            loss = bce + iouL

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f"           Validation Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_sod_model.pth')
        print("           ✔ Saved best model")






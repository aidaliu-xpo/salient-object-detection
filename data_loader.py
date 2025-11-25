import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SaliencyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        image_files = os.listdir(image_dir)

        self.ids = []

        for file in image_files:
            if file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                file_id = os.path.splitext(file)[0]
                self.ids.append(file_id)

        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):


        #get ID like "0003"
        id = self.ids[index]

        #load image and ensure it's RGB (3 channels)
        image_path = f"{self.image_dir}/{id}.jpg"
        image = Image.open(image_path).convert('RGB')

        #load mask and convert it to gray scale (1 channel)
        mask_path = f"{self.mask_dir}/{id}.png"
        mask = Image.open(mask_path).convert('L')

        #convert them to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        #normalize image 0-1 (so the network learns better) and convert mask to either a 0 or 1
        image_np = image_np.astype("float32") / 255.0
        mask_np = (mask_np > 128).astype(np.uint8)

        #add channel to mask -> (1, H, W) -- needed because pytorch expects a channel dimension
        mask_np = np.expand_dims(mask_np, axis=0)

        #reorder image from HWC -> CHW
        image_np = np.transpose(image_np, (2, 0, 1))

        if self.transform is not None:
            image_np, mask_np = self.transform(image_np, mask_np)

                #convert to torch tensors
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        return image_tensor, mask_tensor


class Augment:
    def __init__(self, flip_prob = 0.5, rot_prob=0.3, brightness_prob=0.3):
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.brightness_prob = brightness_prob

    def __call__(self, image_np, mask_np):
        #image_np (3, H, W)
        #mask_np (1, H, W)

        #Horizontal + vertical flip
        if random.random() < self.flip_prob:
            image_np = np.flip(image_np, axis=2).copy() #flip width
            mask_np = np.flip(mask_np, axis=2).copy()

        if random.random() < self.flip_prob:
            image_np = np.flip(image_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=1).copy()

        #rotations (90-degree steps)
        if random.random() < self.rot_prob:
            k = random.choice([1, 2, 3]) #rotate 90/180/270
            image_np = np.rot90(image_np, k = k, axes=(1, 2)).copy()
            mask_np = np.rot90(mask_np, k=k, axes=(1, 2)).copy()

        #brightness jitter
        if random.random() < self.brightness_prob:
            factor = random.uniform(0.8, 1.2)
            image_np = image_np * factor
            image_np = np.clip(image_np, 0.0, 1.0) #keep valid pixel range

        return image_np, mask_np





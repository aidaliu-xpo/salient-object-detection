import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#Picking an image and it's mask
resized_image_path = "data/ECSSD/resized_images_224/0003.jpg"
resized_mask_path = "data/ECSSD/resized_masks_224/0003.png"

#Loading the image
img = Image.open(resized_image_path).convert('RGB')
img_np = np.array(img)
print("Image shape:", img_np.shape)

#Loading the mask
mask = Image.open(resized_mask_path).convert('L') #convert('L') makes the mask single-channel (grayscale) instead of RGB
mask_np = np.array(mask)
print("Mask shape:", mask_np.shape)


print(f"Mask minimum value {mask_np.min()} and maximum value {mask_np.max()}")

#now we need to convert it to binary instead of 0-255

if mask_np.max() > 1:
    mask_np = (mask_np > 128).astype(np.uint8)
    print(f"Converted mask max/min: {mask_np.min()} - {mask_np.max()}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
overlay = img_np.copy()
overlay[mask_np == 1] = [255, 0, 0]
plt.imshow(overlay)
plt.axis("off")

plt.show()
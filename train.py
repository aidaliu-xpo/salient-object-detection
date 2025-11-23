from data_loader import SaliencyDataset
import random
from torch.utils.data import DataLoader, Subset

dataset = SaliencyDataset("data/ECSSD/resized_images_128",
                          "data/ECSSD/resized_masks_128")

N = dataset.__len__()

indexes = [i for i in range(N)]
random.seed(31)
random.shuffle(indexes)

train_end = int(0.70 * N)
val_end = int(0.85 * N)

train_ids = indexes[:train_end]
val_ids = indexes[train_end:val_end]
test_ids = indexes[val_end:]


train_dataset = Subset(dataset, train_ids)
val_dataset = Subset(dataset, val_ids)
test_dataset = Subset(dataset, test_ids)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)


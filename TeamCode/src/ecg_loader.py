import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):

    def __init__(self, dataset, batch_size, crop=128, augment=False):
        super(ECGDataset, self).__init__()

        self.augment = augment
        self.dataset = dataset
        self.p = int(crop // 2)
        self.batch_size = batch_size
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tif, mask = self.dataset[index]
        if len(tif.shape) == 3:#convert to grayscale
            tif = np.mean(tif, axis=2)
        if len(mask.shape) == 3:#reduce to 2D
            mask = np.mean(mask, axis=2)
        pad = ((self.p, self.p), (self.p, self.p))
        tif_pad = np.pad(tif, pad, mode='constant')
        mask_pad = np.pad(mask, pad, mode='constant')

        points = np.where(mask_pad == 1)
        
        points = np.array([[r, c] for r, c in zip(*points)])

        permut = np.random.permutation(np.arange(len(points)))[:self.batch_size]
        batch_points = points[permut]

        patches, masks = [], []
        for point in batch_points:
            r, c = point
            patch = tif_pad[r - self.p:r + self.p,
                            c - self.p:c + self.p]
            mask = mask_pad[r - self.p:r + self.p,
                            c - self.p:c + self.p]

            if self.augment and np.random.uniform() > 0.5:
                choice = np.random.choice(3)
                if choice == 0:
                    patch = np.fliplr(patch)
                    mask = np.fliplr(mask)
                elif choice == 1:
                    patch = np.flipud(patch)
                    mask = np.flipud(mask)
                else:
                    patch = np.rot90(patch, 2)
                    mask = np.rot90(mask, 2)
            patch = np.expand_dims(patch, 0)
            patches.append(patch)
            masks.append(mask)

        patches = np.stack(patches, axis=0)
        masks = np.stack(masks, axis=0)
        return torch.FloatTensor(patches), \
            torch.LongTensor(masks)


class ECGDataLoader(object):

    def __init__(self, ecgs, batch_size, crop=128, augment=False):
        self.crop = crop
        self.ecgs = ecgs
        self.augment = augment
        self.batch_size = batch_size
        return

    def build(self):
        dataset = ECGDataset(self.ecgs, self.batch_size, self.crop, self.augment)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                num_workers=3, pin_memory=True)
        return dataloader

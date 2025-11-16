from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
import torch
import os
import numpy as np

def time_warp(seq, warp_factor_range=(0.8, 1.2), max_frames=60):
    warp = np.random.uniform(*warp_factor_range)
    new_length = int(seq.shape[0] * warp)

    idxs = np.linspace(0, seq.shape[0]-1, new_length).astype(int)
    seq = seq[idxs]

    if len(seq) > max_frames:
        seq = seq[:max_frames]
    elif len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), seq.shape[1]))
        seq = np.vstack([seq, pad])

    return seq

class ASLDataset(Dataset):
    def __init__(self, keypoint_dir, label_dir, augment=False):
        self.keypoint_dir = keypoint_dir
        self.label_dir = label_dir
        self.augment = augment

        self.samples = sorted([f[:-4] for f in os.listdir(keypoint_dir) if f.endswith(".npy")])

        raw_labels = []
        for base in self.samples:
            with open(os.path.join(label_dir, base + ".txt")) as f:
                raw_labels.append(f.read().strip())

        unique = sorted(set(raw_labels))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
        self.targets = [self.label_to_idx[lbl] for lbl in raw_labels]
        print("Label → Index mapping:", self.label_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]
        x = np.load(os.path.join(self.keypoint_dir, base + ".npy")).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            x = time_warp(x)

        x = torch.from_numpy(x).float()
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

class ASLDataModule:
    def __init__(self, keypoint_dir="../data/keypoints", label_dir="../data/labels",
                 batch_size=32, num_workers=4, val_split=0.15, test_split=0.10):
        self.keypoint_dir = keypoint_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self):
        full_ds = ASLDataset(self.keypoint_dir, self.label_dir)

        total = len(full_ds)
        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        train_size = total - val_size - test_size

        train_idx, val_idx, test_idx = random_split(
            range(total),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=True)
        self.train_set.samples = [full_ds.samples[i] for i in train_idx.indices]
        self.train_set.targets = [full_ds.targets[i] for i in train_idx.indices]
        self.train_set.label_to_idx = full_ds.label_to_idx

        self.val_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=False)
        self.val_set.samples = [full_ds.samples[i] for i in val_idx.indices]
        self.val_set.targets = [full_ds.targets[i] for i in val_idx.indices]
        self.val_set.label_to_idx = full_ds.label_to_idx

        self.test_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=False)
        self.test_set.samples = [full_ds.samples[i] for i in test_idx.indices]
        self.test_set.targets = [full_ds.targets[i] for i in test_idx.indices]
        self.test_set.label_to_idx = full_ds.label_to_idx

    def train_dataloader(self):
        class_counts = torch.bincount(torch.tensor(self.train_set.targets))
        class_weights = 1.0 / class_counts.float()
        sample_weights = [class_weights[t] for t in self.train_set.targets]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

from torch.utils.data import DataLoader, random_split, Dataset
import torch
import os
import numpy as np

def time_warp(seq, warp_factor_range=(0.8, 1.2), max_frames=60):
    """
    Randomly speeds up or slows down a sequence by warping the time axis.
    warp < 1.0 → slower (more frames)
    warp > 1.0 → faster (fewer frames)
    """

    warp = np.random.uniform(*warp_factor_range)
    new_length = int(seq.shape[0] * warp)

    # Generate new frame indices
    idxs = np.linspace(0, seq.shape[0]-1, new_length).astype(int)
    seq = seq[idxs]

    # Crop or pad back to fixed length
    if len(seq) > max_frames:
        seq = seq[:max_frames]
    elif len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), seq.shape[1]))
        seq = np.vstack([seq, pad])

    return seq

class ASLDataset(Dataset): 
    """
    Loads ASL .npy keypoint sequences and .txt labels.
    """
    def __init__(self, keypoint_dir, label_dir, augment = False):
        self.keypoint_dir = keypoint_dir
        self.label_dir = label_dir
        self.augment = augment

        # Collect all video base names: video1, video2, ...
        self.samples = sorted([
            f[:-4] for f in os.listdir(keypoint_dir) if f.endswith(".npy")
        ])

        # Load labels
        raw_labels = []
        for base in self.samples:
            with open(os.path.join(label_dir, base + ".txt")) as f:
                raw_labels.append(f.read().strip())

        # Build label → index mapping
        unique = sorted(set(raw_labels))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique)}

        # Convert labels to integers
        self.targets = [self.label_to_idx[lbl] for lbl in raw_labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        # Load sequence
        x = np.load(os.path.join(self.keypoint_dir, base + ".npy")).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:          # 50% chance to augment
                x = time_warp(x)

        x = torch.from_numpy(x)              # (60, 222)

        # Load label index
        y = torch.tensor(self.targets[idx], dtype=torch.long) # Needs to be int64 for CrossEntropy

        return x, y

class ASLDataModule:
    """
    Prepares and loads ASL keypoint sequences
    for training, validation, and testing.
    """
    def __init__(
        self,
        keypoint_dir="../data/keypoints",
        label_dir="../data/labels",
        batch_size=32,
        num_workers=4,
        val_split=0.15,
        test_split=0.10,
    ):
        self.keypoint_dir = keypoint_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self):
        # Load full dataset
        ds = ASLDataset(self.keypoint_dir, self.label_dir)

        total = len(ds)
        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        train_size = total - val_size - test_size

        train_set, val_set, test_set = random_split(
            ds,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # TRAIN: enable augment
        self.train_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=True)
        self.train_set.samples = [ds.samples[i] for i in train_set.indices]
        self.train_set.targets = [ds.targets[i] for i in train_set.indices]

        # VAL/TEST: fresh datasets without augment
        self.val_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=False)
        self.val_set.samples = [ds.samples[i] for i in val_set.indices]
        self.val_set.targets = [ds.targets[i] for i in val_set.indices]

        self.test_set = ASLDataset(self.keypoint_dir, self.label_dir, augment=False)
        self.test_set.samples = [ds.samples[i] for i in test_set.indices]
        self.test_set.targets = [ds.targets[i] for i in test_set.indices]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
# dataloader.py
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
import torch
import os
import numpy as np

# ------- CONFIG: match this to your model -------
SEQ_LEN = 60
FEATURE_DIM = 222   # <-- ensure this matches the feature vector your model uses
# -------------------------------------------------

def time_warp(seq, warp_factor_range=(0.8, 1.2), max_frames=SEQ_LEN):
    warp = np.random.uniform(*warp_factor_range)
    new_length = max(1, int(seq.shape[0] * warp))

    idxs = np.linspace(0, seq.shape[0]-1, new_length).astype(int)
    seq = seq[idxs]

    # pad/truncate frames
    if seq.shape[0] > max_frames:
        seq = seq[:max_frames]
    elif seq.shape[0] < max_frames:
        pad = np.tile(seq[-1], (max_frames - seq.shape[0], 1))
        seq = np.vstack([seq, pad])

    return seq


def jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise


def scale(seq, scale_range=(0.9, 1.1)):
    s = np.random.uniform(*scale_range)
    return seq * s


def translate(seq, shift_range=(-0.05, 0.05)):
    shift = np.random.uniform(*shift_range, size=(1, seq.shape[1]))
    return seq + shift


def frame_dropout(seq, drop_prob=0.10):
    mask = np.random.rand(seq.shape[0]) > drop_prob
    if mask.sum() == 0:
        # if everything dropped, return original sequence padded/truncated to SEQ_LEN
        if seq.shape[0] >= SEQ_LEN:
            return seq[:SEQ_LEN]
        else:
            pad = np.tile(seq[-1], (SEQ_LEN - seq.shape[0], 1))
            return np.vstack([seq, pad])

    seq = seq[mask]

    # pad if needed
    if seq.shape[0] < SEQ_LEN:
        pad = np.tile(seq[-1], (SEQ_LEN - seq.shape[0], 1))
        seq = np.vstack([seq, pad])
    elif seq.shape[0] > SEQ_LEN:
        seq = seq[:SEQ_LEN]
    return seq


def landmark_dropout(seq, drop_prob=0.05):
    mask = np.random.rand(*seq.shape) > drop_prob
    return seq * mask


def augment(seq):
    # augmentation assumes seq already is (SEQ_LEN, FEATURE_DIM)
    if np.random.rand() < 0.5:
        seq = time_warp(seq)

    if np.random.rand() < 0.5:
        seq = jitter(seq)

    if np.random.rand() < 0.3:
        seq = scale(seq)

    if np.random.rand() < 0.3:
        seq = translate(seq)

    if np.random.rand() < 0.2:
        seq = frame_dropout(seq)

    if np.random.rand() < 0.1:
        seq = landmark_dropout(seq)

    # ensure shape after augmentation
    if seq.shape[0] < SEQ_LEN:
        pad = np.tile(seq[-1], (SEQ_LEN - seq.shape[0], 1))
        seq = np.vstack([seq, pad])
    if seq.shape[0] > SEQ_LEN:
        seq = seq[:SEQ_LEN]

    if seq.shape[1] < FEATURE_DIM:
        pad = np.zeros((SEQ_LEN, FEATURE_DIM - seq.shape[1]), dtype=np.float32)
        seq = np.hstack([seq, pad])
    if seq.shape[1] > FEATURE_DIM:
        seq = seq[:, :FEATURE_DIM]

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

    def __len__(self):
        return len(self.samples)

    def _fix_shape(self, arr):
        """
        Ensure arr is numpy array with shape (SEQ_LEN, FEATURE_DIM).
        Handles cases where arr may be (T, F) with T != SEQ_LEN or F != FEATURE_DIM.
        """
        arr = np.array(arr, dtype=np.float32)

        # If 1D vector (rare), try to reshape
        if arr.ndim == 1:
            # assume it's (SEQ_LEN * someF) flatten; fallback to zeros
            try:
                arr = arr.reshape((-1, arr.shape[0] // SEQ_LEN))
            except Exception:
                # fallback: make zeros
                arr = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

        # fix frames (rows)
        if arr.shape[0] < SEQ_LEN:
            pad_frames = np.tile(arr[-1], (SEQ_LEN - arr.shape[0], 1))
            arr = np.vstack([arr, pad_frames])
        elif arr.shape[0] > SEQ_LEN:
            arr = arr[:SEQ_LEN, :]

        # fix features (cols)
        if arr.shape[1] < FEATURE_DIM:
            pad_cols = np.zeros((SEQ_LEN, FEATURE_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad_cols])
        elif arr.shape[1] > FEATURE_DIM:
            arr = arr[:, :FEATURE_DIM]

        return arr

    def __getitem__(self, idx):
        base = self.samples[idx]
        path = os.path.join(self.keypoint_dir, base + ".npy")
        x = np.load(path).astype(np.float32)  # shape may vary: (T, F)

        # Make shape stable BEFORE augmentation:
        x = self._fix_shape(x)

        if self.augment:
            x = augment(x)

        x = torch.from_numpy(x).float()  # shape: (SEQ_LEN, FEATURE_DIM)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


class ASLDataModule:
    def __init__(self, keypoint_dir="../data/keypoints", label_dir="../data/labels",
                 batch_size=32, num_workers=0, val_split=0.15, test_split=0.10):
        # note: num_workers=0 is safer on Windows; set >0 on Linux if desired
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

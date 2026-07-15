"""
Objective ii — Dataset loader for ABIDE structural MRI axial slices.
Loads pre-extracted 2D .nii slices, matches them to DX_GROUP labels
from the ABIDE phenotypic CSV, and prepares them for ResNet50.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


def build_mri_label_table(nii_dir, phenotypic_csv, sub_id_col="SUB_ID", dx_col="DX_GROUP"):
    """
    Scans nii_dir for .nii files, matches each subject ID to its DX_GROUP
    label from the phenotypic CSV. Returns a DataFrame with columns:
    ['sub_id', 'filepath', 'label'] where label: 1 = ASD/autism, 0 = control.

    ABIDE convention: DX_GROUP 1 = Autism, DX_GROUP 2 = Control.
    We remap to match the Kaggle ASD_traits convention: 1 = ASD, 0 = No ASD.
    """
    pheno = pd.read_csv(phenotypic_csv)
    pheno[sub_id_col] = pheno[sub_id_col].astype(int)

    rows = []
    for fname in os.listdir(nii_dir):
        if not fname.endswith(".nii"):
            continue
        try:
            sub_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue

        match = pheno[pheno[sub_id_col] == sub_id]
        if match.empty:
            continue

        dx = match.iloc[0][dx_col]
        label = 1 if dx == 1 else 0  # ABIDE: 1=autism -> our 1=ASD, 2=control -> our 0

        rows.append({
            "sub_id": sub_id,
            "filepath": os.path.join(nii_dir, fname),
            "label": label
        })

    df = pd.DataFrame(rows)
    print(f"✅ Matched {len(df)} MRI slices to phenotypic labels")
    print(f"   Class balance -> ASD (1): {(df['label']==1).sum()}  |  Control (0): {(df['label']==0).sum()}")
    return df


class ABIDESliceDataset(Dataset):
    """
    Loads 2D axial MRI slices and returns them as 3-channel tensors
    (replicated grayscale) sized for ResNet50, normalized to ImageNet stats.
    """
    def __init__(self, dataframe, target_size=224, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.augment = augment
        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _load_slice(self, filepath):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)

        # squeeze any singleton dims, ensure 2D
        data = np.squeeze(data)
        if data.ndim != 2:
            # if a 3D volume slipped through, take the middle slice
            mid = data.shape[-1] // 2
            data = data[..., mid]

        # resize if needed
        if data.shape != (self.target_size, self.target_size):
            from scipy.ndimage import zoom
            zoom_factors = (self.target_size / data.shape[0], self.target_size / data.shape[1])
            data = zoom(data, zoom_factors, order=1)

        # normalize intensity to [0, 1]
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        return data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slice_2d = self._load_slice(row["filepath"])

        # replicate grayscale -> 3 channels for ResNet50
        img_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=0)  # (3, H, W)

        # simple augmentation: random horizontal flip
        if self.augment and np.random.rand() > 0.5:
            img_3ch = img_3ch[:, :, ::-1].copy()

        # ImageNet normalization
        img_3ch = (img_3ch - self.mean[:, None, None]) / self.std[:, None, None]

        img_tensor = torch.tensor(img_3ch, dtype=torch.float32)
        label_tensor = torch.tensor(row["label"], dtype=torch.float32)

        return img_tensor, label_tensor
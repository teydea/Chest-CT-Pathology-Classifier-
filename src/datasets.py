import nibabel as nib
import numpy as np
import os
import glob
from torchvision import transforms
import pydicom
from PIL import Image

import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, paths, labels, ct_format):
        self.paths = paths
        self.labels = labels
        self.ct_format = ct_format

    def __len__(self):
        return len(self.paths)

    def _load_and_preprocess_nii(self, ct_path):
        nii = nib.load(ct_path)
        ct_array = nii.get_fdata()
    
        ct_array = np.clip(ct_array, -1000, 0)
        ct_array = (ct_array + 1000) / 1000
        
        return torch.from_numpy(ct_array).permute(2, 0, 1).unsqueeze(1).float()
        
    def _load_and_preprocess_dcm(self, series_dir):
        dcm_paths = glob.glob(os.path.join(series_dir, "*.dcm"))
        if not dcm_paths:
            dcm_paths = glob.glob(os.path.join(series_dir, "*", "*.dcm"))
    
        if not dcm_paths:
            raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    
        slices = []
        positions = []
        for fp in dcm_paths:
            try:
                ds = pydicom.dcmread(fp, force=True)
                if not hasattr(ds, 'ImagePositionPatient') or ds.ImagePositionPatient is None:
                    continue
                z_pos = float(ds.ImagePositionPatient[2])
                slices.append(ds)
                positions.append(z_pos)
            except Exception as e:
                print(f"Ошибка чтения {fp}: {e}")
                continue
    
        if not slices:
            raise ValueError(f"No valid DICOM slices in {series_dir}")
    
        sorted_pairs = sorted(zip(positions, slices), key=lambda x: x[0])
        sorted_slices = [s for _, s in sorted_pairs]
    
        volume = np.stack([
            s.pixel_array.astype(np.float32) * float(getattr(s, 'RescaleSlope', 1)) +
            float(getattr(s, 'RescaleIntercept', 0))
            for s in sorted_slices
        ], axis=-1)
    
        volume = np.clip(volume, -1000, 0)
        volume = (volume + 1000) / 1000.0
        volume = np.clip(volume, 0, 1)
    
        tensor = torch.from_numpy(volume).permute(2, 0, 1).unsqueeze(1).float()  # [N, H, W]
        return tensor

    def _sample_central_slices(self, ct_array, max_slices=256):
        N = ct_array.shape[0]
        if N <= max_slices:
            return ct_array
        start = (N - max_slices) // 2
        return ct_array[start:start + max_slices]

    def __getitem__(self, idx):
        ct_path = self.paths[idx]
        label = self.labels[idx]
        ct_format = self.ct_format[idx]
    
        if ct_format == 'nii':
            ct_array = self._load_and_preprocess_nii(ct_path)
        elif ct_format == 'dcm':
            ct_array = self._load_and_preprocess_dcm(ct_path)
        else:
            raise ValueError('Неизвестный формат')

        ct_array = self._sample_central_slices(ct_array, max_slices=100)
    
        N, C, H, W = ct_array.shape

        resized_slices = []
        for i in range(N):
            slice_resized = transforms.Resize((384, 384))(ct_array[i])
            resized_slices.append(slice_resized)

        ct_array = torch.stack(resized_slices, dim=0)
        ct_array = ct_array.repeat(1, 3, 1, 1)
    
        return ct_array, torch.tensor(label)    

class SliceDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def _load_and_preprocess_img(self, path):
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
            ])
            image = default_transform(image)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = self._load_and_preprocess_img(img_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label
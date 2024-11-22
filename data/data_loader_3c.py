import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import os
from tqdm import tqdm
import pickle as pkl

class NiftiDataset(Dataset):
    """
    Custom Dataset to load NIfTI files and their labels.
    """
    def __init__(self, csv_file, base_dir=".", mean=86.62818647464137, std=174.31429622300445, transform=None):
        # Load the CSV containing file paths and labels
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir

        # self.file_paths = file_paths
        self.mean = mean
        self.std = std
        self.transform = transform

        split = "metadata"
        if not os.path.exists(f"processed_3c_{split}.pkl"):
            self.full_data = self.preprocess()
            with open(f'processed_3c_{split}.pkl', 'wb') as file:
                pkl.dump(self.full_data, file)
        else:
            with open(f'processed_3c_{split}.pkl', 'rb') as file:
                self.full_data = pkl.load(file)


    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def preprocess(self):
        full_data = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            # row = self.data.iloc[idx]
            rel_file_path = row["File Path"]
            file_path = os.path.join(self.base_dir, rel_file_path)
            label = row["Group"]

            # file_path = self.file_paths[idx]
            img = nib.load(file_path).get_fdata()
            img = (img - self.mean) / self.std
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

            if self.transform:
                img = self.transform(img)

            label_encoded = 0 if label == "CN" else (2 if label == "MCI" else 1)
            label_tensor = torch.tensor(label_encoded, dtype=torch.long)

            full_data.append((img, label_tensor))
        return full_data

    def __getitem__(self, idx):
        return self.full_data[idx]

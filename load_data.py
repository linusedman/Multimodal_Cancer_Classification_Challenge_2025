import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MultiModalCellDataset(Dataset):
    def __init__(self, bf_dir, fl_dir, csv_file=None, transform=None, mode="both"):
        """
        Args:
            bf_dir (str): Path to BF images.
            fl_dir (str): Path to FL images.
            csv_file (str): Path to CSV with 'Name' and 'Diagnosis'. If None, operates in test mode.
            transform (callable, optional): Transform to apply.
            mode (str): 'BF', 'FL', or 'both'
        """
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform
        self.mode = mode.lower()
        self.has_labels = csv_file is not None

        assert self.mode in ['bf', 'fl', 'both'], "mode must be 'BF', 'FL', or 'both'"

        if self.has_labels:
            self.data_df = pd.read_csv(csv_file)
            self.filenames = self.data_df['Name'].tolist()
            self.labels = self.data_df['Diagnosis'].tolist()
        else:
            # For test mode, infer filenames from BF directory
            self.filenames = sorted(os.listdir(bf_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]

        # Load BF and/or FL
        if self.mode in ['bf', 'both']:
            bf_path = os.path.join(self.bf_dir, img_name)
            bf_img = Image.open(bf_path).convert("RGB")
            if self.transform:
                bf_img = self.transform(bf_img)

        if self.mode in ['fl', 'both']:
            fl_path = os.path.join(self.fl_dir, img_name)
            fl_img = Image.open(fl_path).convert("RGB")
            if self.transform:
                fl_img = self.transform(fl_img)

        # Combine
        if self.mode == 'both':
            image = torch.cat([bf_img, fl_img], dim=0)  # (6, H, W)
        elif self.mode == 'bf':
            image = bf_img
        else:
            image = fl_img

        if self.has_labels:
            label = self.labels[idx]
            return image, label
        else:
            return image, img_name  # No label — return filename for test

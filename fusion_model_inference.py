import os
import csv
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
default_loader = __import__('torchvision.datasets.folder', fromlist=['default_loader']).default_loader
from PIL import Image
import pandas as pd
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset for paired BF/FL images ---
class PairedCancerTestDataset(Dataset):
    def __init__(self, bf_folder, fl_folder, transform=None):
        # List and sort filenames present in both
        bf_files = sorted([f for f in os.listdir(bf_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        fl_files = sorted([f for f in os.listdir(fl_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        # Intersection to ensure pairs
        self.names = sorted(list(set(bf_files) & set(fl_files)))
        self.bf_folder = bf_folder
        self.fl_folder = fl_folder
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        bf_path = os.path.join(self.bf_folder, name)
        fl_path = os.path.join(self.fl_folder, name)
        bf_img = Image.open(bf_path).convert("RGB")
        fl_img = Image.open(fl_path).convert("RGB")
        if self.transform:
            bf_img = self.transform(bf_img)
            fl_img = self.transform(fl_img)
        return bf_img, fl_img, name

# --- Fusion model definition (same as training) ---
class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # instantiate backbones with dummy head size
        self.bf_backbone = timm.create_model('swin_large_patch4_window12_384', pretrained=False, num_classes=num_classes)
        self.fl_backbone = timm.create_model('swin_large_patch4_window12_384', pretrained=False, num_classes=num_classes)
        # remove heads i.e. strip the classifier from the OG model
        # We no longer need this!!
        self.bf_backbone.head = nn.Identity()
        self.fl_backbone.head = nn.Identity()
        # fusion head
        # determine feature size dynamically
        with torch.no_grad():
            dummy = torch.randn(1,3,384,384)
            f1 = self.bf_backbone(dummy)
            f1 = f1.view(1, -1, f1.size(-1)).mean(1)
        feat = f1.shape[1]
        self.fusion_head = nn.Sequential(
            nn.Linear(feat*2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, bf, fl):
        f1 = self.bf_backbone(bf)
        f2 = self.fl_backbone(fl)
        # mean pooling
        f1 = f1.view(f1.size(0), -1, f1.size(-1)).mean(1)
        f2 = f2.view(f2.size(0), -1, f2.size(-1)).mean(1)
        f = torch.cat([f1, f2], dim=1)
        return self.fusion_head(f)

# --- Main inference routine ---
def main():
    # Paths
    bf_folder = 'data/BF/test'
    fl_folder = 'data/FL/test'
    fusion_ckpt = 'best_fusion_model.pth'
    output_csv = 'fusion_probs.csv'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # Dataset + Loader
    dataset = PairedCancerTestDataset(bf_folder, fl_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = FusionModel(num_classes=2)
    model = model.to(device)
    # If referencing fusion head directly, ensure attribute name matches
    state_dict = torch.load(fusion_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    with torch.no_grad():
        for bf, fl, names in tqdm(loader, desc='Inference'):
            bf, fl = bf.to(device), fl.to(device)
            logits = model(bf, fl)
            probs = F.softmax(logits, dim=1)
            for name, p in zip(names, probs[:,0].cpu()):  # class 0 probability
                results.append({'Name': name, 'Diagnosis': float(p)})

    # Save DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == '__main__':
    main()

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

# 1) Prepare transforms & loader
test_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485,0.456,0.406],
      std =[0.229,0.224,0.225]
    )
])
class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = sorted([
            os.path.join(folder_path,f) 
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        self.transform = transform
        self.loader    = default_loader
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img      = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

# 2) Load model
class SwinTransformerModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
        in_f = self.model.head.in_features
        self.model.head = nn.Linear(in_f, num_classes)
    def forward(self,x):
        return self.model(x)

num_classes = 2
model = SwinTransformerModel(num_classes).to(device)
ckpt = torch.load('checkpoints/swin_large_patch4_window12_384_finetuned_model.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 3) Inference + softmax
dataset = FlatImageDataset('/srv/scratch1/swallace/CancerSeg/data/BF/test',
                           transform=test_transform)
loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

results = []
counter =0

with torch.no_grad():
    for imgs, fnames in tqdm(loader, desc="Running inference", unit="batch"):
        imgs = imgs.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            raw_logits = model(imgs)                     # [B, 12, 12, 2]
            raw_logits_flat = raw_logits.view(imgs.size(0), -1, raw_logits.size(-1))  # [B, 144, 2]
            logits = raw_logits_flat.mean(dim=1)         # [B, 2]
            probs = F.softmax(logits, dim=1)             # [B, 2]

            if counter == 0:
                print(f"Raw logits shape           : {raw_logits.shape}")
                print(f"After flattening            : {raw_logits_flat.shape}")
                print(f"After mean-pooling (logits) : {logits.shape}")
                print(f"Softmax output (probs)      : {probs.shape}")
                counter += 1

        p_cancer = probs[:, 0]                # ✅ P(Cancerous) is at index 0
        sum_probs = probs.sum(dim=1)          # Should be ≈ 1.0

        for fname, p1, s in zip(fnames, p_cancer.cpu(), sum_probs.cpu()):
            assert abs(s.item() - 1.0) < 1e-4, f"Probabilites must sum to 1: {s.item():.6sf} for image{fname}"
            results.append({'Name': fname, 'Diagnosis': float(p1)})

#
# 4) Build DF, ensure correct order & save
df = pd.DataFrame(results)
assert len(df)==59040, f"Expected 59040 rows but got {len(df)}"
df.to_csv('Epoch12_BF.csv', index=False)
print("Saved submission.csv with", len(df), "rows")
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 1) Prepare transforms & loader
test_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
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
class SwinV2Model(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Model, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window16-256",
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Needed if changing head size
        )

    def forward(self, x):
        # x is a batch of images in [B, C, H, W]
        # Convert to expected input format
        inputs = self.processor(images=[transforms.ToPILImage()(img.cpu()) for img in x], return_tensors="pt", padding=True)
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

num_classes = 2
model = SwinV2Model(num_classes).to(device)
ckpt = torch.load('checkpoints/swin_large_patch4_window12_384_finetuned_model.pth')
model.model.load_state_dict(ckpt['model_state_dict'])
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
            raw_logits = model(imgs)                     # [B, 2]
            probs = F.softmax(raw_logits, dim=1)             # [B, 2]

            if counter == 0:
                print(f"Raw logits shape           : {raw_logits.shape}") 
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
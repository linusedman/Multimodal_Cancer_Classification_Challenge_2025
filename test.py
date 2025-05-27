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
dataset = FlatImageDataset('/srv/scratch1/swallace/CancerSeg/data/FL/test',
                           transform=test_transform)
loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

results = []
count = 0
import torch.nn.functional as F

results = []
counter = 0

with torch.no_grad():
    for imgs, fnames in loader:
        imgs = imgs.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            
                
            raw_logits = model(imgs)            # [B, H, W, 2] SWIN outputs lots of token predictions (16 x 12 x 12 x 2)
            raw_logits_flat = raw_logits.view(raw_logits.size(0), -1, raw_logits.size(-1))
            logits     = raw_logits_flat.mean(dim=1) # [B, 2] We want to average these across image predicitons to just our classes
            probs      = F.softmax(logits, dim=1)  # [B, 2] Compute softmax to get (B * [P(C == 0), P(C == 1)])
            
            
            if(counter == 0):
                print(f"Raw logits shape {raw_logits.shape}")
                print(f"Raw logits shape after flattening {raw_logits_flat.shape}")
                print(f"Logits shape after averaging {logits.shape}")
                print(f"Probs shape {probs.shape}")
                

        p_cancer = probs[:, 1]               # [B] we now have b length probs
        sum_probs = probs.sum(dim=1)         # [B], should be ≈1.0

        for fname, p1, s in zip(fnames, p_cancer.cpu(), sum_probs.cpu()):
            counter += 1
            # print every 1000th image
            if counter % 1000 == 0:
                print(f"[Image #{counter:5d} {fname}]  P0+P1 = {s:.6f}")
    
            results.append({'Name': fname, 'Diagnosis': float(p1)})

#
# 4) Build DF, ensure correct order & save
df = pd.DataFrame(results)
assert len(df)==59040, f"Expected 59040 rows but got {len(df)}"
df.to_csv('submission.csv', index=False)
print("Saved submission.csv with", len(df), "rows")
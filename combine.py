import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import timm
from tqdm import tqdm
import pandas as pd
import os
import csv

bf_paths = glob.glob("data/BF_dataset/Cancerous/*.jpg") \
         + glob.glob("data/BF_dataset/Clean/*.jpg")
fl_paths = glob.glob("data/FL_dataset/Cancerous/*.jpg") \
         + glob.glob("data/FL_dataset/Clean/*.jpg")

# then pass bf_paths instead of bf_folder…
class PairedCancerDataset(Dataset):
    def __init__(self, bf_paths, fl_paths, labels_df, transform=None):
        self.labels = labels_df.set_index("Name")["Diagnosis"].to_dict()
        for i, (k, v) in enumerate(self.labels.items()):
            print(f"  {k} → {v}")
            if i >= 4:
                break
        # build a mapping name -> full path
        self.bf_map = {os.path.basename(p): p for p in bf_paths}
        self.fl_map = {os.path.basename(p): p for p in fl_paths}
        self.names  = sorted(self.labels.keys())
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        bf   = Image.open(self.bf_map[name]).convert("RGB")
        fl   = Image.open(self.fl_map[name]).convert("RGB")
        if self.transform:
            bf = self.transform(bf)
            fl = self.transform(fl)
        label = torch.tensor(self.labels[name], dtype=torch.long)
        return bf, fl, label
    
class FusionModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        
        self.bf_backbone = timm.create_model(
            'swin_large_patch4_window12_384',     # <— the architecture name
            pretrained=False,                    # don't load ImageNet weights
            num_classes=num_classes              # temporarily match your head size
        )
        self.fl_backbone = timm.create_model(
            'swin_large_patch4_window12_384',
            pretrained=False,
            num_classes=num_classes
        )
        
        ckpt_fl = torch.load("/srv/scratch1/swallace/CancerSeg/checkpoints/SWIN_FL_20_EPOCH.pth", map_location="cpu")["model_state_dict"]
        ckpt_bf = torch.load("/srv/scratch1/swallace/CancerSeg/checkpoints/SWINv1_BF_14_EPOCH.pth", map_location="cpu")["model_state_dict"]
        
        self.fl_backbone.load_state_dict(ckpt_fl, strict = False)
        self.bf_backbone.load_state_dict(ckpt_bf, strict = False)
    
        
        for p in self.bf_backbone.parameters(): p.requires_grad = False
        for p in self.fl_backbone.parameters(): p.requires_grad = False
        self.bf_backbone.head = nn.Identity()
        self.fl_backbone.head = nn.Identity()

        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            bf_out = self.bf_backbone(dummy_input)
            fl_out = self.fl_backbone(dummy_input)
        bf_out_flat = bf_out.view(bf_out.size(0), -1, bf_out.size(-1)).mean(dim=1)
        fl_out_flat = fl_out.view(fl_out.size(0), -1, fl_out.size(-1)).mean(dim=1)

        in_feat = bf_out_flat.shape[1]

        print("[Model Info]")
        print(f"  BF raw output shape : {bf_out.shape}")
        print(f"  FL raw output shape : {fl_out.shape}")
        print(f"  BF flattened shape  : {bf_out_flat.shape}")
        print(f"  FL flattened shape  : {fl_out_flat.shape}")
        print(f"  -> using {in_feat * 2} fusion input features (2 × {in_feat})")

        self.fusion_head = nn.Sequential(
            nn.Linear(in_feat * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, bf, fl):
        f_bf = self.bf_backbone(bf)
        f_fl = self.fl_backbone(fl)
        f_bf = f_bf.view(f_bf.size(0), -1, f_bf.size(-1)).mean(dim=1)
        f_fl = f_fl.view(f_fl.size(0), -1, f_fl.size(-1)).mean(dim=1)
        f = torch.cat([f_bf, f_fl], dim=1)
        return self.fusion_head(f)

device = "cuda"
model = FusionModel(num_classes=2).to(device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"✅ Using {torch.cuda.device_count()} GPUs via DataParallel")
    print("POWER !!!")
    model = nn.DataParallel(model)

fusion_head = model.module.fusion_head if isinstance(model, nn.DataParallel) else model.fusion_head
print(f"\n[Summary] Model fusion head input features: {fusion_head[0].in_features}")
print(f"[Summary] Model fusion head output features: {fusion_head[-1].out_features}")

optimizer = optim.Adam(fusion_head.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# transforms + DataLoader
tf = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

labels_df = pd.read_csv('/srv/scratch1/swallace/CancerSeg/data/train.csv')
train_df, val_df = train_test_split(labels_df, test_size=0.1, stratify=labels_df["Diagnosis"])




train_ds = PairedCancerDataset(bf_paths, fl_paths, train_df, transform=tf)
val_ds = PairedCancerDataset(bf_paths, fl_paths, val_df, transform=tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

num_epoch = 10
best_val_acc = 0.0

# Initialize CSV logger
log_path = "fusion_train_log.csv"
with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

for epoch in range(1,num_epoch + 1):
    model.train()
    running_loss = 0.0
    running_correct=  0
    total = 0
    
    for bf, fl, labels in tqdm(train_loader, desc=f'[Train] Epoch {epoch}/{num_epoch}'):
        bf, fl, labels = bf.to(device), fl.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(bf, fl)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * bf.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += bf.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    print(f"    Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    #val
    
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for bf, fl, labels in tqdm(val_loader, desc=f"[ Val] Epoch {epoch}/{num_epoch}"):
            bf, fl, labels = bf.to(device), fl.to(device), labels.to(device)
            outputs = model(bf, fl)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * bf.size(0)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += bf.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc  = val_corrects / val_total
    print(f"  Val   Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

    # Log metrics to CSV
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc])

    # 5) optionally save best model
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "best_fusion_model.pth")
        print(f"  -> new best! saved to best_fusion_model.pth")

print("Training complete. Best val acc: ", best_val_acc)
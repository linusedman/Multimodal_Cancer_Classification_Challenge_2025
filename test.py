import os
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast
from torchvision import datasets, transforms
from PIL import Image
import timm

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

# --- Custom image loader ---
def custom_loader(path):
    img = Image.open(path)
    img = img.convert('RGBA')
    return img.convert('RGB')

# --- Transform used during testing ---
test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Swin Transformer model wrapper ---
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.loader = default_loader  # same as used in ImageFolder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.loader(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.image_paths[idx])
    
# --- Load test dataset ---
test_dataset = FlatImageDataset('/srv/scratch1/swallace/CancerSeg/data/FL/test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- Load model ---
num_classes = 2
model = SwinTransformerModel(num_classes).to(device)
checkpoint = torch.load('checkpoints/swin_large_patch4_window12_384_finetuned_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Run inference and collect logits ---
logit_results = []

with torch.no_grad():
    for i, (inputs, file_names) in enumerate(test_loader):
        inputs = inputs.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1, outputs.size(-1))
            outputs = outputs.mean(dim=1)

        for j in range(inputs.size(0)):
            file_name = file_names[j]
            logits = outputs[j].detach().cpu().numpy().tolist()
            logit_results.append({'filename': file_name, 'logits': logits})

# --- Save logits to CSV ---
df = pd.DataFrame(logit_results)
df.to_csv('submission_logits.csv', index=False)
print("Logits saved to submission_logits.csv")
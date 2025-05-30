import os
import csv
import torch
import pandas as pd
from torch import nn
from torch.amp import autocast
from torchvision import transforms
from PIL import Image
import timm
import glob
from torch.utils.data import Dataset  # assuming it's reusable for inference

# --- Device setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- Image Transform (same as tes  t_transform) ---
test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## Needed we have no labels!
class InferenceImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # returning path as "identifier"

# --- ConvNeXt Model Wrapper ---
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtModel, self).__init__()
        self.model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier and last stage
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True
        

    def forward(self, x):
        return self.model(x)

# --- Load dataset for inference (reusing MultiModalCellDataset in inference mode) ---

image_paths = sorted(glob.glob('data/BF/test/*.jpg'))
inference_dataset = InferenceImageDataset(image_paths, transform=test_transform)

inference_loader = torch.utils.data.DataLoader(
    inference_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# --- Load trained model checkpoint ---
num_classes = 2
model = ConvNeXtModel(num_classes).to(device)
checkpoint = torch.load('checkpoints/convnext_model_full_train.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Run inference ---
logit_results = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(inference_loader):
        inputs = inputs.to(device)

        with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            outputs = model(inputs)  # shape: (batch_size, num_classes)

        for j in range(inputs.size(0)):
            idx = i * inference_loader.batch_size + j
            file_path = inference_loader.dataset.image_paths[idx]
            file_name = os.path.basename(file_path)
            probabilities = torch.softmax(outputs[j], dim=0).detach().cpu().numpy().tolist()
            logit_results.append({'Name': file_name, 'Diagnosis': probabilities[1]})

# --- Save to CSV ---
df = pd.DataFrame(logit_results)
df.to_csv('submission_probs_convnext.csv', index=False)
print("Probabilities saved to submission_probs_convnext.csv")
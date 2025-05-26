import os
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast
from torchvision import datasets, transforms
from PIL import Image
import timn

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
        self.model = timn.create_model('swin_large_patch4_window12_384', pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# --- Load test dataset ---
test_dataset = datasets.ImageFolder('butterflies_dataset', transform=test_transform, loader=custom_loader)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

# --- Load model ---
num_classes = len(test_dataset.classes)
model = SwinTransformerModel(num_classes).to(device)
checkpoint = torch.load('checkpoints/swin_large_patch4_window12_384_finetuned_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Run inference and collect logits ---
logit_results = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1, outputs.size(-1))
            outputs = outputs.mean(dim=1)

        for j in range(inputs.size(0)):
            file_path, _ = test_loader.dataset.samples[i * test_loader.batch_size + j]
            file_name = os.path.basename(file_path)
            logits = outputs[j].detach().cpu().numpy().tolist()
            logit_results.append({'filename': file_name, 'logits': logits})

# --- Save logits to CSV ---
df = pd.DataFrame(logit_results)
df.to_csv('submission_logits.csv', index=False)
print("Logits saved to submission_logits.csv")
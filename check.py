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

class SwinV2Model(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Model, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window16-256",
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Needed if changing head size
        )

model = SwinV2Model(num_classes=2)
ckpt = torch.load("checkpoints/swin_large_patch4_window12_384_finetuned_model.pth", map_location="cpu")
model.load_state_dict(ckpt['model_state_dict'])  # or ckpt directly if only state_dict saved

print(model.classifier)
import os
from datetime import datetime
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torchvision import transforms
import timm
from load_data import MultiModalCellDataset  
from tqdm import tqdm

#GPU else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#Model
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtModel, self).__init__()
        #Different model types to choose between: convnext_tiny, convnext_small, convnext_base, convnext_large
        self.model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True
        for param in self.model.stages[3].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((384, 384)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Main training script
if __name__ == "__main__":

    metrics_file = open("metrics_full.csv", mode="w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "loss", "accuracy", "learning_rate"])

    mp.set_start_method('spawn', force=True)
    print(f"Using device: {device}")

    full_dataset = MultiModalCellDataset(
        bf_dir='data/BF/train',
        fl_dir='data/FL/train',
        csv_file='data/train.csv',
        transform=None,
        mode='BF' #Change between FL and BF
    )

    #Randomly select x number of images - else just choose full_dataset to perform on all images
    random.seed(42)
    subset_indices = random.sample(range(len(full_dataset)), 100)
    base_dataset = full_dataset
    base_dataset.transform = train_transform

    #Train/test-split
    indices = list(range(len(base_dataset)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, test_indices = indices[:split], indices[split:]

    train_dataset = Subset(base_dataset, train_indices)
    test_dataset = Subset(base_dataset, test_indices)

    #Different transform for test/train
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    #Weighted sampling
    targets = [base_dataset[i][1] for i in train_indices]
    class_counts = np.bincount(targets, minlength=max(targets)+1)
    class_weights = 1. / np.where(class_counts == 0, 1, class_counts)  
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=True)

    #Model
    num_classes = len(set(targets)) 
    model = ConvNeXtModel(num_classes).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    checkpoint_path = 'checkpoints/convnext_model_full_train_stage_3.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    counter = 0

    #Training
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate after epoch {epoch+1}: {current_lr:.6f}")

        metrics_writer.writerow([epoch + 1, epoch_loss, epoch_acc, current_lr])

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            print(f"No improvement in loss. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    metrics_file.close()

    #Evaluation
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    test_acc = 100 * correct / total
    avg_test_loss = running_loss / len(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print("Model saved successfully.")
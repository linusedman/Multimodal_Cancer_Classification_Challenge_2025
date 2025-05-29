import os
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

#GPU else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#Model
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtModel, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
        #Freezing of all layers except final
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
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
    mp.set_start_method('spawn', force=True)
    print(f"Using device: {device}")

    full_dataset = MultiModalCellDataset(
        bf_dir='data/BF/train',
        fl_dir='data/FL/train',
        csv_file='data/train.csv',
        transform=None,
        mode='fl'
    )

    #Randomly select x number of images - else just choose full_dataset to perform on all images
    random.seed(42)
    subset_indices = random.sample(range(len(full_dataset)), 100)
    base_dataset = Subset(full_dataset, subset_indices)
    base_dataset.dataset.transform = train_transform

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
    class_weights = 1. / np.where(class_counts == 0, 1, class_counts)  # Undvik division med 0
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint_path = 'checkpoints/convnext_model.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    #Training
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
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

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    #Evaluation
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
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
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
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

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
    writer = SummaryWriter(log_dir="runs/convnext_full_train")

    mp.set_start_method('spawn', force=True)
    print(f"Using device: {device}")

    full_dataset = MultiModalCellDataset(
        bf_dir='data/BF/train',
        fl_dir='data/FL/train',
        csv_file='data/train.csv',
        transform=None,
        mode='BF' #Change between FL and BF
    )

    #Map patient ID to image indices
    patient_to_indices = defaultdict(list)
    patient_to_label = {}

    for idx, filename in enumerate(full_dataset.filenames):
        label = full_dataset.labels[idx]  
        patient_id = filename.split('_')[1] 

        patient_to_indices[patient_id].append(idx)
        patient_to_label[patient_id] = label

    healthy_patients = [pid for pid, label in patient_to_label.items() if label == 0]
    sick_patients = [pid for pid, label in patient_to_label.items() if label == 1]

    #Hardcode one healthy and one sick into validation set
    val_patients = [healthy_patients[0], sick_patients[0]]

    #All others into training set
    train_patients = [pid for pid in patient_to_indices if pid not in val_patients]

    train_indices = [i for pid in train_patients for i in patient_to_indices[pid]]
    val_indices = [i for pid in val_patients for i in patient_to_indices[pid]]

    #Separate dataset instances with separate transforms
    train_dataset_full = MultiModalCellDataset(
        bf_dir='data/BF/train',
        fl_dir='data/FL/train',
        csv_file='data/train.csv',
        transform=train_transform,
        mode='BF'
    )
    val_dataset_full = MultiModalCellDataset(
        bf_dir='data/BF/train',
        fl_dir='data/FL/train',
        csv_file='data/train.csv',
        transform=test_transform,
        mode='BF'
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # #Randomly select x number of images - else just choose full_dataset to perform on all images
    # random.seed(42)
    # subset_indices = random.sample(range(len(full_dataset)), 100)
    # base_dataset = full_dataset
    # base_dataset.transform = train_transform

    # #Train/test-split
    # indices = list(range(len(base_dataset)))
    # random.seed(42)
    # random.shuffle(indices)
    # train_indices = indices

    # # Validation split from training data (90% train / 10% val)
    # val_split = int(0.9 * len(train_indices))
    # final_train_indices = train_indices[:val_split]
    # val_indices = train_indices[val_split:]

    # final_train_dataset = Subset(base_dataset, final_train_indices)
    # val_dataset = Subset(base_dataset, val_indices)

    # final_train_dataset.dataset.transform = train_transform
    # val_dataset.dataset.transform = test_transform

    #Different transform for test/train
    # train_dataset.dataset.transform = train_transform
    # test_dataset.dataset.transform = test_transform

    #Weighted sampling
    targets = [train_dataset_full.labels[i] for i in train_indices]
    class_counts = np.bincount(targets, minlength=max(targets)+1)
    class_weights = 1. / np.where(class_counts == 0, 1, class_counts)  
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=4, sampler=WeightedRandomSampler(
    #     [class_weights[train_dataset_full[i][1]] for i in train_indices],
    #     num_samples=len(train_indices),
    #     replacement=True
    # ), num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

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

        # Validation
        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        print(f"Validation - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.2f}%")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)

        metrics_writer.writerow([epoch + 1, epoch_loss, epoch_acc, current_lr])

        # Early stopping logic
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
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
    writer.close()

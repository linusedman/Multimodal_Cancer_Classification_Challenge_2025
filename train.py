import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import cuda
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
import concurrent.futures
import timm
import csv
from tqdm import tqdm

device = "cuda:0"

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
        # Freezing of all layers except final        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)
    
def custom_loader(path):
    # Open the image
    img = Image.open(path)
    # Ensure the image is in RGB mode even if RGBA
    img = img.convert('RGBA')
    return img.convert('RGB')


train_transform = transforms.Compose([
    transforms.RandomRotation(40),  # Random rotation within a range of 40 degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.Resize((384, 384)),  # Resize the image to 384x384
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),  # Random affine transformations
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),  # Random zoom (rescaling) between 80%-100% of the original size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image with standard mean and std
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device(device)

torch.backends.cudnn.benchmark = True

def load_checkpoint(model, optimizer, checkpoint_path):
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint(['epoch'])
        
        return model, optimizer, epoch
    else:
        print("Checkpoint does not exist")
        return model, optimizer, 0
    

# Main training script
if __name__ == "__main__":
    
    log_file = "training_log.csv"
    with open(log_file, mode = "w", newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # Set the start method for multiprocessing to 'spawn' to avoid potential issues in certain environments
    mp.set_start_method('spawn', force=True)

    # Load dataset and split into train, val, and test sets using random_split
    full_dataset = datasets.ImageFolder('data/FL_dataset', transform=train_transform, loader=custom_loader)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # Create data loaders for training and testing, using batch size of 16, shuffle training data, and use 1 worker for loading
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    num_classes = 2 # Number of output classes based on dataset
    print(f"We have {num_classes} classes")
    model = SwinTransformerModel(num_classes).to(device)  # Initialize the Swin Transformer model

    # Set up Adam optimizer for model parameters with learning rate of 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    # Define CrossEntropyLoss as the loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient scaler for mixed precision training to reduce memory usage and improve speed
    scaler = GradScaler()

    # Set learning rate scheduler to decrease learning rate by a factor of 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Load model checkpoint if available to resume training from the saved state
    checkpoint_path = 'checkpoints/swin_large_patch4_window12_384_finetuned_model.pth'
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Set number of epochs to train for
    epochs = 20

    # Start training loop with ThreadPoolExecutor to parallelize certain tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for epoch in range(start_epoch, epochs):  # Start from the last saved epoch

            # Set model to training mode
            model.train()

            # Initialize running loss and accuracy metrics for the epoch
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over batches in the training data
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}"), 1):
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU or CPU)

                # Zero out gradients from previous step
                optimizer.zero_grad()

                # Use mixed precision to accelerate computation and reduce memory usage
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)  # Forward pass through the model
                    outputs = outputs.view(outputs.size(0), -1, outputs.size(-1))  # Reshape outputs
                    outputs = outputs.mean(dim=1)  # Take the mean across sequence dimension
                    loss = criterion(outputs, labels)  # Compute the loss

                # Backpropagate with scaled gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Clip gradients to prevent explosion and ensure stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Accumulate running loss and accuracy statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)  # Total number of samples
                correct += (predicted == labels).sum().item()  # Count correct predictions

                # Print batch loss and accuracy every 10 batches
                if i % 10 == 0:
                    batch_loss = running_loss / i
                    batch_acc = 100 * correct / total
    

            # Compute and print the epoch loss and accuracy after each epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        outputs = outputs.view(outputs.size(0), -1, outputs.size(-1))
                        outputs = outputs.mean(dim=1)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

            # Update learning rate according to the scheduler
            scheduler.step()

            # Save the model checkpoint at the end of each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    # Switch the model to evaluation mode for testing
    model.eval()

    # Initialize variables for calculating test accuracy
    correct = 0
    total = 0

    # Disable gradient calculation during testing for efficiency
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU or CPU)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)  # Forward pass through the model
                outputs = outputs.view(outputs.size(0), -1, outputs.size(-1))  # Reshape outputs
                outputs = outputs.mean(dim=1)  # Take the mean across sequence dimension
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate and print the test accuracy
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")

    # Notify the user that the model has been saved successfully
    print("Model saved successfully!")
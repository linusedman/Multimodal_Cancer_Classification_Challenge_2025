import os
from PIL import Image
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import cuda
from torch.amp import autocast, GradScaler
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.multiprocessing as mp
import concurrent.futures
import timm
import csv
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __getitem__(self, index):
            img, label = self.dataset[index]
            img = self.transform(img)
            return img, label
        def __len__(self):
            return len(self.dataset)

class SwinV2Model(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Model, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window16-256",
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Needed if changing head size
        )
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, x):
        # x is a batch of images in [B, C, H, W]
        # Convert to expected input format
        inputs = self.processor(images=[transforms.ToPILImage()(img.cpu()) for img in x], return_tensors="pt", padding=True)
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        return self.model(**inputs).logits
    
def custom_loader(path):
    # Open the image
    img = Image.open(path)
    # Ensure the image is in RGB mode even if RGBA
    img = img.convert('RGBA')
    return img.convert('RGB')


train_transform = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = torch.device(device)

torch.backends.cudnn.benchmark = True

def load_checkpoint(model, optimizer, checkpoint_path):
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        return model, optimizer, epoch
    else:
        print("Checkpoint does not exist")
        return model, optimizer, 0
    

# Main training script
if __name__ == "__main__":
    
    log_file = "SWINv2_BF_log.csv"
    with open(log_file, mode = "w", newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # Set the start method for multiprocessing to 'spawn' to avoid potential issues in certain environments
    mp.set_start_method('spawn', force=True)

    # Load dataset and split into train and val sets
    data_set_path = 'data/BF_dataset'
    print(f"Using dataset {data_set_path}")
    full_dataset = datasets.ImageFolder(data_set_path, loader=custom_loader)

    train_size = int(0.001 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformWrapper(train_subset, train_transform)
    val_dataset   = TransformWrapper(val_subset, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    num_classes = 2 # Number of output classes based on dataset
    print(f"We have {num_classes} classes")
    model = SwinV2Model(num_classes).to(device)  # Initialize the Swin Transformer model
    
    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Some info
    net = model.module if isinstance(model, nn.DataParallel) else model

    total_params = 0
    trainable_params = 0
    print("=== Parameter status (requires_grad) ===")
    for name, param in net.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        print(f"{name:60s} : requires_grad={param.requires_grad:5} ({num} params)")
    print("========================================")
    print(f"Total parameters        : {total_params}")
    print(f"Trainable parameters    : {trainable_params}")
    print(f"Frozen parameters       : {total_params - trainable_params}")
    print("========================================")

    # Set up Adam optimizer for model parameters with learning rate of 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    # Define CrossEntropyLoss as the loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient scaler for mixed precision training to reduce memory usage and improve speed
    scaler = GradScaler()

    # Set learning rate scheduler to decrease learning rate by a factor of 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Load model checkpoint if available to resume training from the saved state
    checkpoint_path = 'checkpoints/swinV2.pth'
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Set number of epochs to train for
    epochs = 20

    # Start training loop with ThreadPoolExecutor to parallelize certain tasks
    start_time = time.time()
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
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
    

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
                for inputs, labels in tqdm(val_loader, desc="Validating"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
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
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)


    # Notify the user that the model has been saved successfully
    total_time = time.time() - start_time
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow('Time', total_time)
    print("Model saved successfully!")
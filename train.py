import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from model import CliffordNet
from networks import cliffordnet_12_2, cliffordnet_12_5, cliffordnet_32_3 
from utils import seed_everything
from hybrid_model import clifford_hybrid_nano

# --- Configuration ---
@dataclass
class TrainingConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 200
    weight_decay: float = 0.1
    num_workers: int = 4 if torch.cuda.is_available() else 0
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Dataset params
    data_root: str = './data'
    random_erasing_prob: float = 0.25
    
    num_classes: int = 100
    patch_size: int = 2
    embed_dim: int = 128
    
    # Checkpoint
    save_path: str = 'cliffordnet_cifar100.pth'

# --- Utils ---
def get_device(device_str: str) -> torch.device:
    print(f"Using Device: {device_str.upper()}")
    return torch.device(device_str)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Data Pipeline ---
def get_transforms(cfg: TrainingConfig):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=cfg.random_erasing_prob)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    return transform_train, transform_test

def get_dataloaders(cfg: TrainingConfig):
    print("Preparing Data...")
    train_transform, test_transform = get_transforms(cfg)
    
    trainset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=test_transform
    )

    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    
    return trainloader, testloader

# --- Training Engine ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

@torch.no_grad()
def evaluate(model, loader, device, epoch, best_acc, save_path='best_model.pth'):

    model.eval()
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f"Epoch {epoch} Test Acc: {acc:.2f}%")
    
    if acc > best_acc:
        print(f"🔥 New record! Accuracy improved from {best_acc:.2f}% to {acc:.2f}%")
        best_acc = acc

    return best_acc


# --- Main Execution ---
def main(enable_cuda=False):

    # 1. Setup
    cfg = TrainingConfig()
    seed_everything()
    device = get_device(cfg.device)
    
    # 2. Data
    trainloader, testloader = get_dataloaders(cfg)
    
    # 3. Model Initialization
    print("Initializing Model...")

    try:
        from networks import cliffordnet_12_2
        model = cliffordnet_12_2(
            num_classes=cfg.num_classes, 
            patch_size=cfg.patch_size, 
            embed_dim=cfg.embed_dim,
            enable_cuda=enable_cuda
        )
    except ImportError:
        print("Warning: model not found, using generic CliffordNet.")
        model = CliffordNet(
            num_classes=cfg.num_classes,
            img_size=32, 
            patch_size=cfg.patch_size, 
            embed_dim=cfg.embed_dim,
            depth=12, 
            enable_cuda=enable_cuda
        )
      
    model = model.to(device)
    
    print(f"Model built. Learnable Parameters: {count_parameters(model):,}")

    # 4. Optimization Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 5. Training Loop
    print(f"Start training for {cfg.epochs} epochs...")
    start_time = time.time()
    best_acc = 0.0 
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, cfg.epochs)
        best_acc = evaluate(model, testloader, device, epoch, best_acc, save_path=cfg.save_path)
        scheduler.step()

    total_time = time.time() - start_time
    print(f"Training Finished. Total time: {total_time/60:.2f} mins")


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="supports CUDA acceleration")
    parser.add_argument('--enable_cuda', action='store_true', help='Whether to enable CUDA acceleration (default: False)')
    args = parser.parse_args()    
    print(f"Enable CUDA acceleration: {args.enable_cuda}")
    main(enable_cuda=args.enable_cuda)
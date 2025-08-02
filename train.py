import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.models import ResNet34_Weights, ViT_B_16_Weights
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

# ========== Config ==========
torch.backends.cudnn.benchmark = True
DATA_DIR = 'data'
FILELIST = os.path.join(DATA_DIR, 'filelist.txt')
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 7
MIXUP_PROB = 0
CUTMIX_PROB = 0
USE_VIT = False  # Set True to use ViT from torchvision
NUM_WORKERS = min(8, os.cpu_count() // 2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== Dataset Class ==========
class SketchDataset(Dataset):
    def __init__(self, filelist, transform=None):
        self.samples = []
        with open(filelist, 'r') as f:
            for line in f:
                path = line.strip()
                label = os.path.basename(os.path.dirname(path))
                self.samples.append((os.path.join(DATA_DIR, path), label))
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(set(x[1] for x in self.samples)))}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_idx[label]

# ========== Mixup / CutMix ==========
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    _, H, W = x.size()[1:]
    rx, ry = np.random.randint(W), np.random.randint(H)
    rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x1, y1 = np.clip(rx - rw // 2, 0, W), np.clip(ry - rh // 2, 0, H)
    x2, y2 = np.clip(rx + rw // 2, 0, W), np.clip(ry + rh // 2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# ========== Early Stopping ==========
class EarlyStopper:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0

    def check(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ========== Training ==========
def train():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomAffine(10, scale=(0.95, 1.05)),
        transforms.ToTensor()
    ])

    dataset = SketchDataset(FILELIST, transform)
    labels = dataset.label_to_idx
    num_classes = len(labels)

    # --- START OF CHANGES ---

    # 1. Calculate class distribution to identify imbalance
    print("Calculating class distribution...")
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[labels[label]] += 1
    
    print("Class Counts:", {list(labels.keys())[i]: class_counts[i] for i in range(num_classes)})

    # 2. Calculate weights for the loss function
    # Weights are the inverse of the class frequency
    total_samples = sum(class_counts)
    class_weights = [total_samples / count if count > 0 else 0 for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    print("Calculated Loss Weights:", class_weights)

    # --- END OF CHANGES ---

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_set = torch.utils.data.Subset(dataset, indices[:split])
    val_set = torch.utils.data.Subset(dataset, indices[split:])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    weights = ViT_B_16_Weights.DEFAULT if USE_VIT else ResNet34_Weights.DEFAULT
    model = models.vit_b_16(weights=weights) if USE_VIT else models.resnet34(weights=weights)
    if USE_VIT:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = nn.DataParallel(model).to(DEVICE)
    
    # --- CHANGE: Apply the calculated weights to the loss function ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3,)
    stopper = EarlyStopper(PATIENCE)
    scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, targets in loop:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            if random.random() < MIXUP_PROB:
                images, y_a, y_b, lam = mixup_data(images, targets)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            elif random.random() < CUTMIX_PROB:
                images, y_a, y_b, lam = cutmix_data(images, targets)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        writer.add_scalar("Loss/Train", total_loss, epoch)

        # ======= Validation =======
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    y_true.extend(targets.cpu())
                    y_pred.extend(preds.cpu())

        acc = accuracy_score(y_true, y_pred)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", acc, epoch)

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'sketch_model_best.pt')

        if stopper.check(acc):
            print("Early stopping triggered.")
            break

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    torch.save(model.state_dict(), 'sketch_model_final.pt')
    print("Final model saved to sketch_model_final.pt")
    writer.close()
    
if __name__ == "__main__":
    train()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "efficientnet_xray.pth")
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 15
LR          = 0.001
VAL_SPLIT   = 0.2  # 20% for validation

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names  = full_dataset.classes
print(f"Classes found: {class_names}")
print(f"Total images : {len(full_dataset)}")

# Train / Validation split
val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Train: {train_size} | Validation: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# MODEL — EfficientNet-B0
# ─────────────────────────────────────────────
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model = model.to(device)
print("\nEfficientNet-B0 loaded successfully!")

# ─────────────────────────────────────────────
# LOSS AND OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
best_val_acc = 0.0

print("\n--- Starting Training ---\n")

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size * 100

    # ── Validate ──
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / val_size * 100

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}% | "
          f"Loss: {train_loss/len(train_loader):.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"   💾 Best model saved! (Val Acc: {val_acc:.1f}%)")

print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.1f}%")
print(f"✅ Model saved at: {MODEL_PATH}")